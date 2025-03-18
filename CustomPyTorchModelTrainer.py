import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer
import logging
import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from typing import Any

logger = logging.getLogger(__name__)

class CustomPyTorchModelTrainer(PyTorchModelTrainer):
    """
    Erweiterter PyTorch-Trainer mit zusätzlicher Ausgabe von Trainingsmetriken
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        data_convertor,
        model_meta_data: dict[str, Any] = {},
        window_size: int = 1,
        tb_logger: Any = None,
        **kwargs
    ):
        # Hier den super().__init__ mit benannten Parametern aufrufen
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            data_convertor=data_convertor,
            model_meta_data=model_meta_data,
            window_size=window_size,
            tb_logger=tb_logger,
            **kwargs
        )

        # Holen Sie den Scheduler aus kwargs, falls vorhanden
        self.scheduler = kwargs.get('scheduler', None)
        self.best_val_loss = float('inf')
        self.early_stop_patience = kwargs.get('patience', 10)
        self.patience_counter = 0

    def _split_data(self, data_dictionary, splits=None):
        """
        Teilt das Daten-Dictionary in Trainings- und Validierungssets auf.
        """
        if splits is None:
            splits = ["train", "test"]

        train_dict = {
            "features": data_dictionary["train_features"],
            "labels": data_dictionary["train_labels"]
        }

        val_dict = {
            "features": data_dictionary.get("test_features", pd.DataFrame()),
            "labels": data_dictionary.get("test_labels", pd.DataFrame())
        }

        return train_dict, val_dict

    def _create_dataset(self, data_dict):
        """
        Erstellt ein PyTorch Dataset aus dem Daten-Dictionary.
        """
        x = self.data_convertor.convert_x(data_dict["features"], device=self.device)
        y = self.data_convertor.convert_y(data_dict["labels"], device=self.device)
        dataset = TensorDataset(x, y)
        return dataset

    def train(self, dataloader, epochs=None, steps=None):
        """
        Überschreibt die train-Methode, um Loss pro Iteration auszugeben
        und Learning Rate Scheduling zu unterstützen
        """
        self.model.train()
        epoch = 0
        step = 0
        running_loss = 0.0
        log_interval = 10  # Alle 10 Schritte ausgeben

        # Trainingsschleife
        while (epochs is None or epoch < epochs) and (steps is None or step < steps):
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                # KORRIGIERT: Verschiebe die Tensoren auf das richtige Gerät ohne sie erneut zu konvertieren
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward-Pass
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)

                # Backward-Pass und Optimierung
                loss.backward()
                self.optimizer.step()

                # Learning Rate Scheduler updaten
                if self.scheduler is not None:
                    self.scheduler.step()

                # Loss akkumulieren
                running_loss += loss.item()

                # Loss pro Iteration ausgeben
                if step % log_interval == 0:
                    avg_loss = running_loss / (log_interval if step > 0 else 1)
                    running_loss = 0.0

                    # Berechne Genauigkeit für Klassifikation
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == y_batch).sum().item() / y_batch.size(0)

                    # logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                    # logger.info(f"Training - E{epoch+1} B{batch_idx+1} (Step {step}) - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

                step += 1
                if steps is not None and step >= steps:
                    break

            epoch += 1
            if epochs is not None and epoch >= epochs:
                break

        return self

    def fit(self, data_dictionary, splits=None):
        """
        Überschreibt die fit-Methode, um Trainingsmetriken auszugeben
        und Early Stopping zu unterstützen
        """
        # Standard-Fit-Methode
        is_early_stop = False

        # Early Stopping initialisieren
        best_model_state = None

        # Daten aufteilen
        train_dict, val_dict = self._split_data(data_dictionary, splits)

        for i in range(self.n_epochs) if self.n_epochs is not None else range(1):
            # Trainings-Dataloader erstellen
            train_dataset = self._create_dataset(train_dict)
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )

            # Eine Epoche Training
            self.train(train_dataloader, epochs=1)

            # Validierung nach jeder Epoche
            self.model.eval()
            with torch.no_grad():
                # Validierungsmetriken
                x_val = self.data_convertor.convert_x(val_dict["features"], device=self.device)
                y_val = self.data_convertor.convert_y(val_dict["labels"], device=self.device)
                y_pred_val = self.model(x_val)

                # Berechne Loss
                val_loss = self.criterion(y_pred_val, y_val).item()

                # Berechne Genauigkeit
                _, predicted = torch.max(y_pred_val, 1)
                val_acc = (predicted == y_val).sum().item() / y_val.size(0)

                # logger.info(f"Epoche {i+1}/{self.n_epochs}, Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

                # Prüfe auf Early Stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stop_patience:
                        logger.info(f"Early stopping nach {i+1} Epochen")
                        is_early_stop = True
                        break

            self.model.train()

            # Breche ab, wenn Early Stopping aktiviert wurde
            if is_early_stop:
                break

        # Lade das beste Modell, wenn Early Stopping verwendet wurde
        if best_model_state is not None and is_early_stop:
            self.model.load_state_dict(best_model_state)

        # Ausgabe der finalen Trainings- und Validierungsmetriken
        self.model.eval()
        with torch.no_grad():
            # Trainingsmetriken
            x_train = self.data_convertor.convert_x(train_dict["features"], device=self.device)
            y_train = self.data_convertor.convert_y(train_dict["labels"], device=self.device)
            y_pred_train = self.model(x_train)

            # Berechne Loss
            train_loss = self.criterion(y_pred_train, y_train).item()

            # Berechne Genauigkeit für Klassifikation
            _, predicted = torch.max(y_pred_train, 1)
            train_acc = (predicted == y_train).sum().item() / y_train.size(0)

            logger.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Testmetriken, falls Testdaten vorhanden
            if len(val_dict["features"]) > 0:
                x_test = self.data_convertor.convert_x(val_dict["features"], device=self.device)
                y_test = self.data_convertor.convert_y(val_dict["labels"], device=self.device)
                y_pred_test = self.model(x_test)

                # Berechne Loss
                test_loss = self.criterion(y_pred_test, y_test).item()

                # Berechne Genauigkeit für Klassifikation
                _, predicted = torch.max(y_pred_test, 1)
                test_acc = (predicted == y_test).sum().item() / y_test.size(0)

                logger.info(f"Validation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

        return self

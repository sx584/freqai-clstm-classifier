import logging
from typing import Any, Dict, List

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import Sequential  # Import Sequential

from freqtrade.freqai.base_models.BasePyTorchClassifier import BasePyTorchClassifier
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import (
    DefaultPyTorchDataConvertor,
    PyTorchDataConvertor,
)
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer
from freqtrade.freqai.torch.CustomPyTorchModelTrainer import CustomPyTorchModelTrainer
from freqtrade.freqai.torch.CLSTMModel import CLSTMModel

# Add both classes to the safe globals list
torch.serialization.add_safe_globals([
    CustomPyTorchModelTrainer,
    CLSTMModel,
    Sequential
])

logger = logging.getLogger(__name__)

'''
Verbesserte Signalstabilisierung:
def predict(self, features_dict, dk: FreqaiDataKitchen, **kwargs):
    """
    Verbesserte predict-Methode mit sicherer Signalstabilisierung (muss noch getestet werden)
    """
    import numpy as np
    import pandas as pd
    import logging

    # Reguläre Vorhersage durchführen
    preds_dict = super().predict(features_dict, dk, **kwargs)

    # Prüfen, ob Signalstabilisierung aktiviert ist
    if not self.model_training_parameters.get('signal_stabilization', False):
        return preds_dict

    # Aktueller Zeitpunkt
    current_time = dk.pair_df['date'].iloc[-1]
    prediction_data = {}

    # Frühere Vorhersagen abrufen
    previous_predictions = dk.get_predictions_to_date(current_time)
    stabilized_count = 0
    total_preds = 0

    for label, pred_df in preds_dict.items():
        total_preds += 1

        # Sicherheitsprüfungen
        if pred_df is None or pred_df.empty or label not in pred_df.columns:
            prediction_data[label] = pred_df
            continue

        # Aktuelle Vorhersage sicher extrahieren
        current_pred = pred_df[label].iloc[0]

        # Prüfen auf frühere Vorhersagen
        if (previous_predictions is not None and
            not previous_predictions.empty and
            label in previous_predictions.columns):

            # Letzte Vorhersage extrahieren
            last_pred = previous_predictions[label].iloc[-1]

            # Stabilisierungsschwellenwert
            threshold = self.model_training_parameters.get('signal_stability_threshold', 0.01)

            # Prüfen, ob sich die Vorhersage signifikant geändert hat
            if abs(float(current_pred) - float(last_pred)) <= threshold:
                # Alte Vorhersage beibehalten
                pred_df_stabilized = pd.DataFrame({label: [float(last_pred)]},
                                                 index=pred_df.index)
                prediction_data[label] = pred_df_stabilized
                stabilized_count += 1
                continue

        # Ohne Stabilisierung: Original-Vorhersage verwenden
        prediction_data[label] = pred_df

    # Debug-Informationen
    if total_preds > 0:
        logging.info(f"Signalstabilisierung: {stabilized_count}/{total_preds} Vorhersagen stabilisiert")

    return prediction_data

'''


class PyTorchCLSTMClassifier(BasePyTorchClassifier):
    """
    CLSTM (Convolutional LSTM) Classifier-Modell für FreqAI.

    Dieses Modell kombiniert CNN- und LSTM-Schichten wie in der Forschungsstudie beschrieben:
    Khattak et al. Journal of Big Data (2024) 11:58

    Das Modell besteht aus:
    1. CNN-Blöcken zur Merkmalsextraktion
    2. LSTM-Schicht zur Erfassung zeitlicher Abhängigkeiten
    3. Dense-Schichten (MLP) für die endgültige Klassifikation

    Die Modellparameter können in der Konfigurationsdatei eingestellt werden:
    {
        "freqai": {
            "model_training_parameters": {
                "learning_rate": 3e-4,
                "trainer_kwargs": {
                    "n_steps": 5000,
                    "batch_size": 64,
                    "n_epochs": null,
                },
                "model_kwargs": {
                    "cnn_blocks": 2,
                    "lstm_units": 64,
                    "lstm_layers": 1,
                    "dense_layers": 2,
                    "dense_neurons": 14,
                    "dropout_percent": 0.2,
                    "use_attention": true
                },
                "use_class_weights": true,
                "class_weights_method": "focused",
                "signal_stabilization": true,
                "hysteresis_value": 0.15
            }
        }
    }
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(
            target_tensor_type=torch.long, squeeze_target_tensor=True
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = config.get("trainer_kwargs", {})
        self.use_class_weights: bool = config.get("use_class_weights", False)
        self.class_weights_method: str = config.get("class_weights_method", "inverse_frequency")

        # Signalstabilisierung
        self.signal_stabilization: bool = config.get("signal_stabilization", False)
        self.hysteresis_value: float = config.get("hysteresis_value", 0.15)
        self.previous_predictions = {}

        # Spezielles Logging für Fokusklassen
        self.focus_classes = ["strong_down", "neutral", "strong_up"]

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Training des CLSTM-Modells für die Klassifikation
        """
        class_names = self.get_class_names()
        self.convert_label_column_to_int(data_dictionary, dk, class_names)

        # Analyse der Klassenverteilung
        labels = data_dictionary["train_labels"]
        unique_labels, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip([class_names[int(l)] for l in unique_labels], counts))
        total = sum(counts)
        percentages = {label: count/total*100 for label, count in distribution.items()}

        logger.info(f"Klassenverteilung (absolut): {distribution}")
        logger.info(f"Klassenverteilung (prozentual): {percentages}")

        # Spezifische Lösung für das FreqAI-Label-Problem
        # Rufe fit_labels() NACH convert_label_column_to_int auf
        dk.fit_labels()

        # Forciere Klasseninitialiserung, falls sie in labels_mean/std fehlen
        for class_name in class_names:
            if class_name not in dk.data["labels_mean"]:
                dk.data["labels_mean"][class_name] = 0.0
            if class_name not in dk.data["labels_std"]:
                dk.data["labels_std"][class_name] = 1.0

        # Bestimme die Anzahl der Features aus den Eingabedaten
        n_features = data_dictionary['train_features'].shape[1]

        # CLSTM-Modell erstellen
        model = CLSTMModel(
            input_dim=n_features,
            output_dim=len(class_names),
            **self.model_kwargs
        )
        model.to(self.device)

        # Optimizer mit Weight Decay für Regularisierung
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        # Scheduler erstellen, falls konfiguriert
        scheduler = None
        if self.trainer_kwargs.get("scheduler", "") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer_kwargs.get("n_steps", 5000),
                eta_min=self.learning_rate / 10
            )
            # Scheduler aus trainer_kwargs entfernen, um Doppelparameter zu vermeiden
            if "scheduler" in self.trainer_kwargs:
                self.trainer_kwargs = {k: v for k, v in self.trainer_kwargs.items() if k != "scheduler"}

        # Klassengewichtung implementieren, falls aktiviert
        if self.use_class_weights:
            class_weights = self._calculate_class_weights(distribution, class_names)
            logger.info(f"Verwendete Klassengewichte: {dict(zip(class_names, class_weights.tolist()))}")
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = torch.nn.CrossEntropyLoss()

        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            # Explizit scheduler als Teil von Kwargs übergeben
            trainer_kwargs = {**self.trainer_kwargs}
            if scheduler is not None:
                trainer_kwargs["scheduler"] = scheduler

            trainer = CustomPyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                model_meta_data={"class_names": class_names},
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **trainer_kwargs,
            )

        trainer.fit(data_dictionary, self.splits)
        return trainer

    def _log_confusion_matrix(self, trainer, data_dictionary, class_names):
        """Erstellt und loggt eine Konfusionsmatrix für das trainierte Modell"""
        trainer.model.eval()
        with torch.no_grad():
            # Validierungsdaten
            if "test_features" in data_dictionary and len(data_dictionary["test_features"]) > 0:
                x_test = self.data_convertor.convert_x(data_dictionary["test_features"], device=self.device)
                y_test = self.data_convertor.convert_y(data_dictionary["test_labels"], device=self.device)

                # Vorhersagen
                y_pred_test = trainer.model(x_test)
                _, predicted = torch.max(y_pred_test, 1)

                # Konvertiere Tensoren zu Numpy-Arrays
                y_true = y_test.cpu().numpy()
                y_pred = predicted.cpu().numpy()

                # Berechne Konfusionsmatrix
                cm = confusion_matrix(y_true, y_pred)

                # Logge Konfusionsmatrix
                cm_str = "\nKonfusionsmatrix:\n"
                cm_str += " " * 15 + " ".join([f"{name[:10]:>10}" for name in class_names]) + "\n"
                for i, row in enumerate(cm):
                    cm_str += f"{class_names[i][:10]:>15} " + " ".join([f"{val:>10}" for val in row]) + "\n"

                logger.info(cm_str)

                # Zusätzliche Metriken für Fokusklassen
                focus_indices = [class_names.index(cls) for cls in self.focus_classes if cls in class_names]
                for idx in focus_indices:
                    true_pos = cm[idx, idx]
                    false_pos = cm[:, idx].sum() - true_pos
                    false_neg = cm[idx, :].sum() - true_pos

                    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    logger.info(f"Klasse {class_names[idx]} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    def _calculate_class_weights(self, distribution, class_names):
        """
        Berechnet Klassengewichte basierend auf der Verteilung und der gewählten Methode
        """
        if self.class_weights_method == "focused":
            # Fokussierte Gewichtung: Bevorzugt starke Klassen und neutral
            total = sum(distribution.values())
            weights = []

            focus_factor = 1.2  # Zusätzlicher Multiplikator für Fokusklassen

            for name in class_names:
                count = distribution.get(name, 1)

                # Grundgewicht (inverse Frequenz mit Dämpfung)
                weight = np.sqrt(total / (len(distribution) * count))

                # Zusätzliche Gewichtung für Fokusklassen
                if name in self.focus_classes:
                    weight *= focus_factor

                weights.append(weight)

            # Normalisieren, damit der Durchschnitt 1.0 ist
            weights_tensor = torch.FloatTensor(weights)
            weights_tensor = weights_tensor / weights_tensor.mean()

            return weights_tensor

        elif self.class_weights_method == "inverse_frequency":
            # Berechne inverse Frequenz als Gewicht
            total = sum(distribution.values())
            weights = []
            for name in class_names:
                count = distribution.get(name, 1)  # Fallback auf 1, falls Klasse nicht in Verteilung
                # Inverse Frequenz: seltenere Klassen bekommen höhere Gewichte
                weights.append(total / (len(distribution) * count))

            # Normalisiere Gewichte
            weights_tensor = torch.FloatTensor(weights)
            weights_tensor = weights_tensor / weights_tensor.sum() * len(weights)

            return weights_tensor

        elif self.class_weights_method == "sqrt_inverse_frequency":
            # Quadratwurzel der inversen Frequenz für mildere Gewichtung
            total = sum(distribution.values())
            weights = []
            for name in class_names:
                count = distribution.get(name, 1)
                weights.append(np.sqrt(total / (len(distribution) * count)))

            weights_tensor = torch.FloatTensor(weights)
            weights_tensor = weights_tensor / weights_tensor.sum() * len(weights)

            return weights_tensor

        elif self.class_weights_method == "balanced":
            # Einfache ausgewogene Gewichtung
            weights = []
            for name in class_names:
                count = distribution.get(name, 1)
                weights.append(1.0 / count)

            weights_tensor = torch.FloatTensor(weights)
            weights_tensor = weights_tensor / weights_tensor.sum() * len(weights)

            return weights_tensor

        else:
            # Fallback auf keine Gewichtung
            logger.warning(f"Unbekannte Gewichtungsmethode: {self.class_weights_method}. Verwende keine Gewichtung.")
            return torch.ones(len(class_names))

    # def predict(self, features_dict, dk: FreqaiDataKitchen, **kwargs):
    #     """
    #     Überschreibt die Predict-Methode, um Signalstabilisierung zu implementieren
    #     """
    #     # Standard-Vorhersage durchführen
    #     predictions = super().predict(features_dict, dk, **kwargs)

    #     if not self.signal_stabilization or dk.pair not in predictions:
    #         return predictions

    #     # Signalstabilisierung mit Hysterese
    #     if dk.pair not in self.previous_predictions:
    #         self.previous_predictions[dk.pair] = None

    #     current_proba = predictions[dk.pair]

    #     # Klassennamen abrufen
    #     class_names = self.get_class_names()
    #     proba_columns = [f"{name}_proba" for name in class_names]

    #     # Identifiziere die aktuelle Vorhersage (Klasse mit höchster Wahrscheinlichkeit)
    #     current_class_probs = {}
    #     for i, name in enumerate(class_names):
    #         if f"{name}_proba" in current_proba:
    #             current_class_probs[name] = current_proba[f"{name}_proba"].iloc[-1].item()

    #     # Aktuelle Klasse mit höchster Wahrscheinlichkeit
    #     if not current_class_probs:  # Sicherheitscheck
    #         return predictions

    #     current_max_class = max(current_class_probs.items(), key=lambda x: x[1])[0]
    #     current_max_prob = current_class_probs[current_max_class]

    #     # Wenn dies die erste Vorhersage ist, speichern und zurückgeben
    #     if self.previous_predictions[dk.pair] is None:
    #         self.previous_predictions[dk.pair] = current_max_class
    #         return predictions

    #     # Rufe die vorherige Vorhersage ab
    #     prev_class = self.previous_predictions[dk.pair]
    #     prev_prob = current_class_probs.get(prev_class, 0)

    #     # Hysterese: Nur wechseln, wenn die Differenz groß genug ist
    #     if current_max_class != prev_class and (current_max_prob - prev_prob) < self.hysteresis_value:
    #         # Wenn die Differenz nicht groß genug ist, behalte die vorherige Klasse bei
    #         logger.debug(f"Stabilisierung: Behalte Klasse {prev_class} bei (statt {current_max_class})")

    #         # Tausche die Wahrscheinlichkeiten aus, um die vorherige Klasse zu bevorzugen
    #         # (höhere Wahrscheinlichkeit für die vorherige Klasse)
    #         for i, name in enumerate(class_names):
    #             prob_col = f"{name}_proba"
    #             if name == prev_class:
    #                 predictions[dk.pair][prob_col].iloc[-1] = current_max_prob
    #             elif name == current_max_class:
    #                 predictions[dk.pair][prob_col].iloc[-1] = prev_prob

    #         # Aktualisiere den Trend-Indikator direkt, falls vorhanden
    #         if "&s-trend_strength" in predictions[dk.pair]:
    #             predictions[dk.pair]["&s-trend_strength"].iloc[-1] = prev_class

    #         # Behalte die vorherige Klasse bei
    #         current_max_class = prev_class

    #     # Speichere die aktuelle Vorhersage für die nächste Iteration
    #     self.previous_predictions[dk.pair] = current_max_class

    #     return predictions

    def predict(self, features_dict, dk: FreqaiDataKitchen, **kwargs):
        """
        Überschreibt die Predict-Methode, um Signalstabilisierung zu implementieren.
        Fügt Debug-Statements ein und stellt sicher, dass numerische Skalare verwendet werden.
        """
        # Standard-Vorhersage durchführen
        predictions = super().predict(features_dict, dk, **kwargs)

        if not self.signal_stabilization or dk.pair not in predictions:
            return predictions

        # Signalstabilisierung mit Hysterese
        if dk.pair not in self.previous_predictions:
            self.previous_predictions[dk.pair] = None

        current_proba = predictions[dk.pair]

        # Klassennamen abrufen
        class_names = self.get_class_names()

        # Identifiziere die aktuelle Vorhersage: Hole den letzten (aktuellen) Wahrscheinlichkeitswert
        current_class_probs = {}
        for name in class_names:
            col_name = f"{name}_proba"
            if col_name in current_proba:
                # Extrahiere den letzten Wert aus der Spalte und logge den rohen Wert sowie dessen Typ
                value = current_proba[col_name].iloc[-1]
                logger.debug(f"Raw prediction for {name}: {value} (type: {type(value)})")
                try:
                    # Explizite Umwandlung in float
                    scalar_value = float(value)
                except Exception as e:
                    logger.error(f"Konvertierung des Werts für {name} zu float fehlgeschlagen: {e}")
                    scalar_value = 0.0
                current_class_probs[name] = scalar_value

        # Sicherheitscheck: Falls keine Wahrscheinlichkeiten gefunden wurden
        if not current_class_probs:
            return predictions

        # Ermittlung der Klasse mit der höchsten Wahrscheinlichkeit
        current_max_class = max(current_class_probs.items(), key=lambda x: x[1])[0]
        current_max_prob = current_class_probs[current_max_class]
        logger.debug(f"Aktuelle maximale Klasse: {current_max_class}, Wahrscheinlichkeit: {current_max_prob}")

        # Erstes Prediction: Speichere die Vorhersage und gebe zurück
        if self.previous_predictions[dk.pair] is None:
            self.previous_predictions[dk.pair] = current_max_class
            return predictions

        # Rufe die vorherige Vorhersage ab
        prev_class = self.previous_predictions[dk.pair]
        prev_prob = current_class_probs.get(prev_class, 0)
        logger.debug(f"Vorherige Klasse: {prev_class}, Wahrscheinlichkeit: {prev_prob}, Hysterese-Wert: {self.hysteresis_value}")

        # Hysterese: Nur wechseln, wenn die Differenz groß genug ist
        if current_max_class != prev_class and (current_max_prob - prev_prob) < self.hysteresis_value:
            logger.debug(f"Stabilisierung: Behalte Klasse {prev_class} bei (statt {current_max_class}). Differenz: {current_max_prob - prev_prob}")

            # Tausche die Wahrscheinlichkeiten aus, um die vorige Klasse zu bevorzugen
            for name in class_names:
                prob_col = f"{name}_proba"
                if name == prev_class:
                    predictions[dk.pair][prob_col].iloc[-1] = current_max_prob
                    logger.debug(f"Setze {prob_col} auf {current_max_prob} für vorherige Klasse.")
                elif name == current_max_class:
                    predictions[dk.pair][prob_col].iloc[-1] = prev_prob
                    logger.debug(f"Setze {prob_col} auf {prev_prob} für aktuelle (nicht gewechselte) Klasse.")

            # Aktualisiere den Trend-Indikator direkt, falls vorhanden
            if "&s-trend_strength" in predictions[dk.pair]:
                predictions[dk.pair]["&s-trend_strength"].iloc[-1] = prev_class
                logger.debug(f"Aktualisiere &s-trend_strength auf {prev_class}")

            # Behalte die vorige Klasse bei
            current_max_class = prev_class

        # Speichere die aktuelle Vorhersage für die nächste Iteration
        self.previous_predictions[dk.pair] = current_max_class

        return predictions

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """
        Überschreibt die Standard-Methode, um sicherzustellen, dass alle Klassenbezeichnungen
        in dk.data["labels_mean"] und dk.data["labels_std"] vorhanden sind.
        """
        import scipy as spy

        # Original FreqAI-Verhalten beibehalten (aus dem regulären FreqAI-Code)
        num_candles = self.freqai_info.get("fit_live_predictions_candles", 100)

        # Initialisiere leere Dictionaries für labels_mean und labels_std
        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}

        # Hole alle verfügbaren Klassen aus dem Modell
        class_names = self.get_class_names()

        # Zuerst standard-mäßige Statistiken für vorhandene Labels berechnen
        # (ähnlich dem Original-Code in freqai_interface.py)
        for label in dk.label_list + dk.unique_class_list:
            if label in self.dd.historic_predictions[dk.pair] and self.dd.historic_predictions[dk.pair][label].dtype != object:
                try:
                    f = spy.stats.norm.fit(self.dd.historic_predictions[dk.pair][label].tail(num_candles))
                    dk.data["labels_mean"][label], dk.data["labels_std"][label] = f[0], f[1]
                except Exception as e:
                    dk.data["labels_mean"][label] = 0.0
                    dk.data["labels_std"][label] = 1.0

        # Dann sicherstellen, dass ALLE Klassenbezeichnungen im Dictionary sind
        for class_name in class_names:
            if class_name not in dk.data["labels_mean"]:
                dk.data["labels_mean"][class_name] = 0.0
            if class_name not in dk.data["labels_std"]:
                dk.data["labels_std"][class_name] = 1.0

        return

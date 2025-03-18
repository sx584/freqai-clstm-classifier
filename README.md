Add PyTorchCLSTMClassifier in user_data/freqaimodels
Add the other two files to your torch-Folder in freqtrade/freqai/torch

Example Config:
"freqai": {
        "enabled": true,
        "activate_tensorboard" : true,
        "purge_old_models": 2,
        "expiration_hours": 4,
        "live_retrain_hours": 2,
        "train_period_days": 20,
        "backtest_period_days": 5,
        "save_backtest_models": true,
        "write_metrics_to_disk": true,
        "identifier": "clstm1",
        "fit_live_predictions_candles": 24,
        "weibull_outlier_threshold": 0.999,
        "optuna_hyperopt": false,
        "track_performance": true,
        "feature_parameters": {
            "include_corr_pairlist": [
                "BTC/USDT:USDT"
            ],
            "include_timeframes": [
                "1m",
                "5m"
            ],
            "label_period_candles": 15,
            "include_shifted_candles": 3,
            // "DI_threshold": 20,
            "weight_factor": 0.5,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": false,
            "use_DBSCAN_to_remove_outliers": false,
            "indicator_periods_candles": [5, 10, 20, 30, 60],
            "noise_standard_deviation": 0.1,
            "buffer_train_data_candles": 25
        },
        "data_split_parameters": {
            "test_size": 0.25,
            "random_state": 187,
            "shuffle": false
        },
        "model_training_parameters": {
            "learning_rate": 1e-3,
            "trainer_kwargs": {
                "n_steps": 5000,
                "batch_size": 16, //64
                "n_epochs": 100,
                "patience": 30,
                "scheduler": "cosine"
            },
            "model_kwargs": {
                "cnn_blocks": 3,
                "lstm_units": 20, //32
                "lstm_layers": 2,
                "dense_layers": 3,
                "dense_neurons": 20, //32
                "dropout_percent": 0.3,
                "use_attention": true
            },
            "use_class_weights": true,
            "class_weights_method": "focused",
            "signal_stabilization": false,
            "hysteresis_value": 0.15
        }
        

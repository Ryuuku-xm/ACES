import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, cohen_kappa_score
import numpy as np
import os


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dl, val_dl, test_dl, device, logger, args, configs,
                 experiment_log_dir, seed):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        self.logger = logger
        self.args = args
        self.configs = configs
        self.experiment_log_dir = experiment_log_dir
        self.seed = seed
        self.patience = configs.patience
        # 初始化每个指标的最佳值为无穷大（最小化）或负无穷大（最大化）
        self.best_val_loss = float('inf')
        self.best_val_mse = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_qwk = -float('inf')  # QWK是最大化指标
        self.early_stopping_counter = 0

    def train(self):
        for epoch in range(self.configs.epoch):
            self.model.train()
            total_loss = 0.0
            all_labels = []
            all_preds = []

            for batch in self.train_dl:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

            avg_train_loss = total_loss / len(self.train_dl)
            train_mse = mean_squared_error(all_labels, all_preds)
            train_rmse = np.sqrt(train_mse)
            train_qwk = cohen_kappa_score(np.round(all_labels).astype(int), np.round(all_preds).astype(int), weights='quadratic')

            self.logger.info(
                f"Epoch {epoch + 1}/{self.configs.epoch}, Train Loss: {avg_train_loss}, MSE: {train_mse}, "
                f"RMSE: {train_rmse}, QWK: {train_qwk}")

            # 验证模型并更新各指标最佳值
            val_loss, val_mse, val_rmse, val_qwk = self.validate()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
            if val_qwk > self.best_val_qwk:  # QWK是最大化指标
                self.best_val_qwk = val_qwk

            # 提前停止检查
            self.early_stopping_counter += 1 if val_loss >= self.best_val_loss else 0
            if self.early_stopping_counter >= self.patience:
                self.logger.info(f"Validation loss did not improve within {self.patience} epochs, stopping early.")
                break

            self.scheduler.step()

            # 输出每个指标的当前最佳验证值
            self.logger.info(
                f"Current Best Validation - Loss: {self.best_val_loss}, MSE: {self.best_val_mse}, "
                f"RMSE: {self.best_val_rmse}, QWK: {self.best_val_qwk}")

    def validate(self):
        self.model.eval()
        total_val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.val_dl:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                val_loss = self.compute_loss(outputs, labels)

                total_val_loss += val_loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().detach().numpy())

        avg_val_loss = total_val_loss / len(self.val_dl)
        val_mse = mean_squared_error(all_labels, all_preds)
        val_rmse = np.sqrt(val_mse)
        val_qwk = cohen_kappa_score(np.round(all_labels).astype(int), np.round(all_preds).astype(int), weights='quadratic')

        self.logger.info(f"Validation Loss: {avg_val_loss}, MSE: {val_mse}, RMSE: {val_rmse}, QWK: {val_qwk}")

        return avg_val_loss, val_mse, val_rmse, val_qwk

    def compute_loss(self, outputs, labels):
        labels = labels.float()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, labels)
        return loss

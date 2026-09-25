# excel_epoch_logger.py
# 记录每个 epoch 的指标并导出为 Excel（不包含 escape_time）

import os
from dataclasses import dataclass, asdict
from typing import List, Optional
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    max_val_acc: float
    total_time: float
    epoch_lambda: float


class ExcelEpochLogger:
    """
    用法：
      logger = ExcelEpochLogger(out_dir)
      logger.log(epoch, train_loss, train_acc, val_loss, val_acc, max_val_acc, total_time)
      logger.save()
    """

    def __init__(self, out_dir: str, filename: str = "epoch_metrics.xlsx", sheet_name: str = "metrics"):
        self.out_dir = out_dir
        self.filename = filename
        self.sheet_name = sheet_name
        self.records: List[EpochRecord] = []

    @property
    def save_path(self) -> str:
        return os.path.join(self.out_dir, self.filename)

    def log(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        max_val_acc: float,
        total_time: float,
        epoch_lambda: float,
    ) -> None:
        self.records.append(
            EpochRecord(
                epoch=int(epoch),
                train_loss=float(train_loss),
                train_acc=float(train_acc),
                val_loss=float(val_loss),
                val_acc=float(val_acc),
                max_val_acc=float(max_val_acc),
                total_time=float(total_time),
                epoch_lambda=float(epoch_lambda),
            )
        )

    def save(self, out_dir: Optional[str] = None) -> str:
        target_dir = out_dir if out_dir is not None else self.out_dir
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, self.filename)

        wb = Workbook()
        ws = wb.active
        ws.title = self.sheet_name

        headers = [
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "max_val_acc",
            "total_time",
            "epoch_lambda",
        ]
        ws.append(headers)

        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=c)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for r in self.records:
            d = asdict(r)
            ws.append([d[h] for h in headers])

        ws.freeze_panes = "A2"
        for col_idx, h in enumerate(headers, start=1):
            max_len = len(h)
            for row_idx in range(2, ws.max_row + 1):
                v = ws.cell(row=row_idx, column=col_idx).value
                if v is not None:
                    max_len = max(max_len, len(str(v)))
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 32)

        wb.save(path)
        return path

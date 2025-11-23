# -*- coding: utf-8 -*-
import os
import sys
import re
import time
from PySide6.QtCore import Qt, Signal, QThread, QSize, QByteArray
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QFrame, QProgressBar, QRadioButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QScrollArea, QFormLayout, QSizePolicy, QMessageBox,
    QGroupBox, QListWidget, QSplitter, QListWidgetItem, QButtonGroup
)
from PySide6.QtSvg import QSvgRenderer

# Corrected relative imports
from .. import settings
from ..app_logic import AppLogic

# --- Constants ---
DEFAULT_FONT_SIZE = 13
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 18

# --- SVG Icons (Base64 Encoded) ---
ICON_INFO = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8+PHBhdGggZD0ibTEyIDE2djBtMC04djQiLz48L3N2Zz4="
ICON_DELETE = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxsaW5lIHgxPSIxOCIgeDI9IjYiIHkxPSI2IiB5Mj0iMTgiLz48bGluZSB4MT0iNiIgeDI9IjE4IiB5MT0iNiIgeTI9IjE4Ii8+PC9zdmc+Cg=="

def create_icon(base64_svg, color="#555555"):
    svg_data = QByteArray.fromBase64(base64_svg.encode('utf-8'))
    svg_str = svg_data.data().decode('utf-8').replace('currentColor', color)
    pixmap = QPixmap(QSize(64, 64))
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer = QSvgRenderer(svg_str.encode('utf-8'))
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)

class FileListItemWidget(QWidget):
    itemDeleted = Signal(QListWidgetItem)
    def __init__(self, text, list_item, parent=None):
        super().__init__(parent)
        self.list_item = list_item
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(10)
        self.label = QLabel(text); self.label.setWordWrap(True)
        self.delete_button = QPushButton(); self.delete_button.setIcon(create_icon(ICON_DELETE, "#888888"))
        self.delete_button.setFixedSize(24, 24); self.delete_button.setObjectName("deleteButton")
        self.delete_button.clicked.connect(self._on_delete)
        layout.addWidget(self.label, 1); layout.addWidget(self.delete_button)
    def _on_delete(self): self.itemDeleted.emit(self.list_item)


class MainWindow(QWidget):
    fontSizeChanged = Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF 内容提取器 Pro")
        app_icon_path = os.path.join(os.path.dirname(__file__), "assets", "app.png")
        if os.path.exists(app_icon_path):
            self.setWindowIcon(QIcon(app_icon_path))
        self.settings = settings.load_settings()
        self.setAcceptDrops(True)
        self.resize(self.settings.get("window_width", 900), self.settings.get("window_height", 650))
        self.worker_thread = None; self.logic = None; self.current_file_index = 0
        self.file_list = []; self.batch_start_time = 0
        self.total_images_extracted = 0; self.total_tables_extracted = 0
        self._init_ui()
        self._connect_signals()
        self._load_and_apply_settings()

    def _init_ui(self):
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)
        toolbar = self._create_toolbar(); main_layout.addWidget(toolbar)
        splitter = QSplitter(Qt.Horizontal); main_layout.addWidget(splitter, 1)
        self.main_content_widget = self._create_main_content_widget(); splitter.addWidget(self.main_content_widget)
        sidebar_widget = self._create_sidebar_widget(); splitter.addWidget(sidebar_widget)
        splitter.setSizes([s if s > 100 else 300 for s in self.settings.get("splitter_sizes", [500, 300])])
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 0)
        status_bar = self._create_status_bar(); main_layout.addWidget(status_bar)

    def _create_toolbar(self):
        toolbar_widget = QWidget(); toolbar_widget.setObjectName("toolbar")
        toolbar_layout = QHBoxLayout(toolbar_widget); toolbar_layout.setContentsMargins(10, 5, 10, 5); toolbar_layout.setSpacing(8)
        self.add_file_button = QPushButton("添加文件"); self.add_folder_button = QPushButton("添加文件夹"); self.clear_list_button = QPushButton("清空列表")
        self.about_button = QPushButton("关于"); self.about_button.setIcon(create_icon(ICON_INFO))
        toolbar_layout.addWidget(self.add_file_button); toolbar_layout.addWidget(self.add_folder_button)
        toolbar_layout.addWidget(self.clear_list_button); toolbar_layout.addStretch(); toolbar_layout.addWidget(self.about_button)
        return toolbar_widget

    def _create_main_content_widget(self):
        widget = QWidget(); layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 15, 20, 15); layout.setSpacing(15)
        self.drop_area_label = QLabel("将一个或多个 PDF 文件拖拽到此区域"); self.drop_area_label.setAlignment(Qt.AlignCenter); self.drop_area_label.setObjectName("dropAreaLabel")
        self.file_list_widget = QListWidget(); layout.addWidget(self.file_list_widget, 1)
        self.original_list_style = "QListWidget { background-color: #f7f7f7; border: 1px dashed #d0d0d0; border-radius: 12px; }"; self.file_list_widget.setStyleSheet(self.original_list_style)
        layout.insertWidget(0, self.drop_area_label); self.drop_area_label.show()
        return widget

    def _create_sidebar_widget(self):
        sidebar_widget = QWidget(); sidebar_widget.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar_widget); sidebar_layout.setContentsMargins(0, 0, 0, 15); sidebar_layout.setSpacing(0)
        settings_scroll_area = QScrollArea(); settings_scroll_area.setWidgetResizable(True); settings_scroll_area.setFrameShape(QFrame.NoFrame)
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container); settings_layout.setContentsMargins(20, 15, 20, 15); settings_layout.setSpacing(20)
        general_group = QGroupBox("通用设置"); general_layout = QFormLayout(general_group); general_layout.setRowWrapPolicy(QFormLayout.WrapLongRows); general_layout.setSpacing(10)
        self.mode_group = QButtonGroup(self); self.mode3_radio = QRadioButton("智能解析"); self.mode_group.addButton(self.mode3_radio); self.mode1_radio = QRadioButton("仅提取图片"); self.mode_group.addButton(self.mode1_radio); self.mode2_radio = QRadioButton("全页转图"); self.mode_group.addButton(self.mode2_radio); self.mode3_radio.setChecked(True)
        mode_radio_layout = QVBoxLayout(); mode_radio_layout.addWidget(self.mode3_radio); mode_radio_layout.addWidget(self.mode1_radio); mode_radio_layout.addWidget(self.mode2_radio); general_layout.addRow("提取模式:", mode_radio_layout)
        self.format_group = QButtonGroup(self); self.png_radio = QRadioButton("png"); self.format_group.addButton(self.png_radio); self.webp_radio = QRadioButton("webp"); self.format_group.addButton(self.webp_radio); self.png_radio.setChecked(True)
        format_radio_layout = QHBoxLayout(); format_radio_layout.addWidget(self.png_radio); format_radio_layout.addWidget(self.webp_radio); format_radio_layout.addStretch(); general_layout.addRow("输出格式:", format_radio_layout)
        self.optimize_webp_checkbox = QCheckBox("优化 WebP 图像"); general_layout.addRow("", self.optimize_webp_checkbox)
        self.font_size_spinbox = QSpinBox(); self.font_size_spinbox.setRange(MIN_FONT_SIZE, MAX_FONT_SIZE); self.font_size_spinbox.setSuffix(" px"); general_layout.addRow("字体大小:", self.font_size_spinbox)
        self.numbers_lineedit = QLineEdit(); self.numbers_lineedit.setPlaceholderText("例: 1,3,5-8 (留空为全部)"); general_layout.addRow("指定序号:", self.numbers_lineedit)
        self.output_dir_lineedit = QLineEdit(); self.browse_button = QPushButton("浏览"); output_dir_layout = QHBoxLayout(); output_dir_layout.addWidget(self.output_dir_lineedit); output_dir_layout.addWidget(self.browse_button); general_layout.addRow("输出目录:", output_dir_layout)
        self.auto_output_dir_checkbox = QCheckBox("为每个PDF创建独立文件夹"); self.auto_output_dir_checkbox.setChecked(True); general_layout.addRow("", self.auto_output_dir_checkbox); settings_layout.addWidget(general_group)
        self.mode3_group_box = QGroupBox("智能解析设置"); mode3_layout = QFormLayout(self.mode3_group_box); mode3_layout.setSpacing(10)
        self.dpi_spinbox_m3 = QSpinBox(); self.dpi_spinbox_m3.setRange(72, 600); self.dpi_spinbox_m3.setSingleStep(50)
        self.conf_spinbox = QDoubleSpinBox(); self.conf_spinbox.setRange(0.1, 1.0); self.conf_spinbox.setSingleStep(0.05)
        self.table_conf_spinbox = QDoubleSpinBox(); self.table_conf_spinbox.setRange(0.1, 1.0); self.table_conf_spinbox.setSingleStep(0.05)
        self.padding_spinbox = QSpinBox(); self.padding_spinbox.setRange(0, 100); self.extract_tables_checkbox = QCheckBox("同时提取表格")
        mode3_layout.addRow("输出 DPI:", self.dpi_spinbox_m3); mode3_layout.addRow("图形置信度:", self.conf_spinbox); mode3_layout.addRow("表格置信度:", self.table_conf_spinbox); mode3_layout.addRow("裁剪边距:", self.padding_spinbox); mode3_layout.addRow("", self.extract_tables_checkbox); settings_layout.addWidget(self.mode3_group_box)
        settings_layout.addStretch()
        settings_scroll_area.setWidget(settings_container); sidebar_layout.addWidget(settings_scroll_area, 1)
        self.start_button = QPushButton("开始提取"); self.start_button.setObjectName("primaryButton"); self.start_button.setEnabled(False); self.start_button.setMinimumHeight(44)
        start_button_container = QWidget(); start_button_layout = QHBoxLayout(start_button_container); start_button_layout.setContentsMargins(20, 0, 20, 0); start_button_layout.addWidget(self.start_button); sidebar_layout.addWidget(start_button_container)
        return sidebar_widget

    def _create_status_bar(self):
        status_bar_widget = QWidget(); status_bar_widget.setObjectName("statusBar")
        status_bar_layout = QHBoxLayout(status_bar_widget); status_bar_layout.setContentsMargins(20, 8, 20, 8)
        self.status_label = QLabel("欢迎使用！"); self.status_label.setObjectName("statusLabel")
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False)
        status_bar_layout.addWidget(self.status_label, 1); status_bar_layout.addWidget(self.progress_bar); self.progress_bar.hide()
        return status_bar_widget

    def _connect_signals(self):
        self.add_file_button.clicked.connect(self._select_pdf_files); self.add_folder_button.clicked.connect(self._select_folder)
        self.clear_list_button.clicked.connect(self._clear_file_list); self.browse_button.clicked.connect(self._select_output_dir)
        self.about_button.clicked.connect(self._show_about_dialog); self.start_button.clicked.connect(self._start_processing)
        self.auto_output_dir_checkbox.toggled.connect(self._toggle_auto_dir)
        self.mode3_radio.toggled.connect(self._update_ui_for_mode)
        self.webp_radio.toggled.connect(self._update_optimize_visibility)
        self.font_size_spinbox.valueChanged.connect(self.fontSizeChanged.emit)

    def _load_and_apply_settings(self):
        self.font_size_spinbox.setValue(self.settings.get("font_size", DEFAULT_FONT_SIZE))
        mode = self.settings.get("mode_text", "智能解析"); "mode 1" in mode and self.mode1_radio.setChecked(True); "mode 2" in mode and self.mode2_radio.setChecked(True)
        self.settings.get("output_format", "png") == "webp" and self.webp_radio.setChecked(True)
        self.optimize_webp_checkbox.setChecked(self.settings.get("optimize_m3", True))
        self.auto_output_dir_checkbox.setChecked(self.settings.get("auto_output_dir", True)); self._toggle_auto_dir(self.settings.get("auto_output_dir", True))
        self.output_dir_lineedit.setText(self.settings.get("output_dir", "output"))
        self.dpi_spinbox_m3.setValue(self.settings.get("dpi", 300)); self.conf_spinbox.setValue(self.settings.get("conf", 0.4))
        self.table_conf_spinbox.setValue(self.settings.get("table_min_conf", 0.25))
        self.padding_spinbox.setValue(self.settings.get("padding", 10)); self.extract_tables_checkbox.setChecked(self.settings.get("extract_tables", True))
        self.numbers_lineedit.setText(self.settings.get("target_numbers", "")); self._update_ui_for_mode(); self._update_optimize_visibility()

    def _gather_current_settings(self):
        if self.mode1_radio.isChecked(): mode_text = "仅提取图片 (mode 1)"
        elif self.mode2_radio.isChecked(): mode_text = "全页转图 (mode 2)"
        else: mode_text = "智能解析 (mode 3)"
        self.settings.update({
            "mode_text": mode_text, "output_format": "webp" if self.webp_radio.isChecked() else "png",
            "auto_output_dir": self.auto_output_dir_checkbox.isChecked(), "output_dir": self.output_dir_lineedit.text(),
            "dpi": self.dpi_spinbox_m3.value(), "conf": self.conf_spinbox.value(), "table_min_conf": self.table_conf_spinbox.value(),
            "padding": self.padding_spinbox.value(), "extract_tables": self.extract_tables_checkbox.isChecked(),
            "target_numbers": self.numbers_lineedit.text(), "optimize_m3": self.optimize_webp_checkbox.isChecked(),
            "font_size": self.font_size_spinbox.value(),
            "imgsz": 1024, "parallel": True, "optimize": self.optimize_webp_checkbox.isChecked()
        })
        return self.settings

    def _update_ui_for_mode(self, checked=False):
        if self.sender() and hasattr(self.sender(), 'isChecked') and not self.sender().isChecked(): return
        self.mode3_group_box.setVisible(self.mode3_radio.isChecked())
        self._update_optimize_visibility()
        
    def _update_optimize_visibility(self): self.optimize_webp_checkbox.setVisible(self.webp_radio.isChecked())
    def _update_list_placeholder(self): self.drop_area_label.setVisible(self.file_list_widget.count() == 0)

    def _add_files_to_list(self, paths):
        current_files = {self.file_list_widget.item(i).data(Qt.UserRole) for i in range(self.file_list_widget.count())}
        added_count = 0
        for path in paths:
            if path not in current_files:
                list_item = QListWidgetItem(self.file_list_widget); list_item.setData(Qt.UserRole, path)
                item_widget = FileListItemWidget(os.path.basename(path), list_item); item_widget.itemDeleted.connect(self._remove_list_item)
                list_item.setSizeHint(item_widget.sizeHint()); self.file_list_widget.addItem(list_item)
                self.file_list_widget.setItemWidget(list_item, item_widget); added_count += 1
        if added_count > 0: self.start_button.setEnabled(True); self.status_label.setText(f"已添加 {self.file_list_widget.count()} 个文件到处理队列。")
        self._update_list_placeholder()

    def _remove_list_item(self, item):
        row = self.file_list_widget.row(item); self.file_list_widget.takeItem(row)
        if self.file_list_widget.count() == 0: self.start_button.setEnabled(False); self.status_label.setText("文件队列已清空。")
        self._update_list_placeholder()

    def _clear_file_list(self):
        self.file_list_widget.clear(); self.start_button.setEnabled(False); self.status_label.setText("文件队列已清空。")
        self._update_list_placeholder()

    def _select_pdf_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择PDF文件", "", "PDF Files (*.pdf)"); paths and self._add_files_to_list(paths)

    def _select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹");
        if path:
            pdf_files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs if f.lower().endswith('.pdf')]
            if pdf_files: self._add_files_to_list(pdf_files)
            else: self.status_label.setText(f"在 '{os.path.basename(path)}' 中未找到PDF。")

    def _toggle_auto_dir(self, checked): self.output_dir_lineedit.setEnabled(not checked); self.browse_button.setEnabled(not checked)
    def _select_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录"); path and self.output_dir_lineedit.setText(path)
        
    def dragEnterEvent(self, event):
        if self.isEnabled() and event.mimeData().hasUrls() and all(url.toLocalFile().lower().endswith('.pdf') for url in event.mimeData().urls()):
            event.acceptProposedAction(); self.file_list_widget.setStyleSheet("background-color: #e6f3ff; border: 2px solid #007aff; border-radius: 12px;")
        else: event.ignore()

    def dragLeaveEvent(self, event): self.file_list_widget.setStyleSheet(self.original_list_style)

    def dropEvent(self, event):
        self.file_list_widget.setStyleSheet(self.original_list_style)
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self._add_files_to_list(files)
        event.acceptProposedAction()

    def _start_processing(self):
        if self.file_list_widget.count() == 0: return self._show_error("请先添加至少一个 PDF 文件。")
        self.file_list = [self.file_list_widget.item(i).data(Qt.UserRole) for i in range(self.file_list_widget.count())]
        self.current_file_index = 0; self.total_images_extracted = 0; self.total_tables_extracted = 0
        self.batch_start_time = time.time()
        self.base_settings = self._gather_current_settings()
        settings.save_settings(self.base_settings) # Save settings on start
        self._set_controls_enabled(False); self.progress_bar.show(); self._process_next_file()

    def _process_next_file(self):
        if self.current_file_index >= len(self.file_list):
            duration = time.time() - self.batch_start_time; self.status_label.setText("批量处理完成！")
            self.progress_bar.setValue(100); self._set_controls_enabled(True)
            QMessageBox.information(self, "任务完成", f"所有 {len(self.file_list)} 个文件处理完毕。\n\n总耗时: {duration:.2f} 秒\n共提取图片: {self.total_images_extracted} 张\n共提取表格: {self.total_tables_extracted} 个")
            self.progress_bar.hide(); [self.file_list_widget.item(i).setFont(QApplication.font()) for i in range(self.file_list_widget.count())]
            return

        pdf_path = self.file_list[self.current_file_index]; file_name = os.path.basename(pdf_path)
        for i in range(self.file_list_widget.count()): self.file_list_widget.item(i).setFont(QApplication.font())
        font = self.file_list_widget.item(self.current_file_index).font(); font.setBold(True); self.file_list_widget.item(self.current_file_index).setFont(font)
        self.status_label.setText(f"处理中 ({self.current_file_index + 1}/{len(self.file_list)}): {file_name}")
        self.progress_bar.setValue(int((self.current_file_index / len(self.file_list)) * 100))
        current_settings = self.base_settings.copy(); current_settings["input_path"] = pdf_path
        if self.auto_output_dir_checkbox.isChecked():
            pdf_dir, _ = os.path.split(pdf_path); pdf_basename, _ = os.path.splitext(file_name)
            current_settings["output_dir"] = os.path.join(pdf_dir, pdf_basename)
        self.logic = AppLogic(); self.worker_thread = QThread(); self.logic.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(lambda: self.logic.run_extraction(current_settings))
        self.logic.progress_updated.connect(self._update_progress); self.logic.status_changed.connect(self._update_status_for_file)
        self.logic.processing_finished.connect(self._on_file_processing_finished); self.worker_thread.start()

    def _on_file_processing_finished(self, result_message):
        images_found = re.search(r"(\d+)\s*张图片", result_message); tables_found = re.search(r"(\d+)\s*个表格", result_message)
        if images_found: self.total_images_extracted += int(images_found.group(1))
        if tables_found: self.total_tables_extracted += int(tables_found.group(1))
        self.worker_thread.quit(); self.worker_thread.wait(); self.logic.deleteLater(); self.worker_thread.deleteLater()
        self.logic = self.worker_thread = None; self.current_file_index += 1; self._process_next_file()

    def _update_progress(self, value):
        if not self.file_list: return
        base = (self.current_file_index / len(self.file_list)) * 100
        self.progress_bar.setValue(int(base + (value / len(self.file_list))))

    def _update_status_for_file(self, message):
        fn = os.path.basename(self.file_list[self.current_file_index]) if self.current_file_index < len(self.file_list) else ""
        self.status_label.setText(f"[{fn}] {message}")

    def _set_controls_enabled(self, enabled):
        self.findChild(QWidget, "toolbar").setEnabled(enabled)
        self.findChild(QSplitter).widget(1).setEnabled(enabled) # Sidebar
        self.start_button.setEnabled(enabled and self.file_list_widget.count() > 0)

    def _show_about_dialog(self): QMessageBox.about(self, "关于本软件", """<h3>PDF 内容提取器 Pro v2.3</h3>...""")
    def _show_error(self, message): QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "任务正在进行", "请等待当前处理任务完成。"); event.ignore()
        else:
            settings_to_save = self._gather_current_settings()
            settings_to_save["window_width"] = self.width(); settings_to_save["window_height"] = self.height()
            settings_to_save["splitter_sizes"] = self.findChild(QSplitter).sizes()
            settings.save_settings(settings_to_save); event.accept()

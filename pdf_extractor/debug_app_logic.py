def _ai_extract_mode_3(self, s: Dict) -> str:
    print("DEBUG: Entered _ai_extract_mode_3")
    # Build a robust path to the default model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_model_path = os.path.join(project_root, 'models', 'doclayout_yolo_docstructbench_imgsz1024.onnx')
    print(f"DEBUG: Default model path constructed: {default_model_path}")
    
    model_path = s.get("model_path") or default_model_path
    print(f"DEBUG: Final model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"DEBUG: Model path does not exist: {os.path.abspath(model_path)}")
        return f"错误: 模型文件不存在于 '{os.path.abspath(model_path)}'"

    self.status_changed.emit("正在加载 ONNX 模型...")
    try:
        print("DEBUG: Loading ONNX model...")
        model = onnx.load(model_path)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        stride = ast.literal_eval(metadata.get("stride", "32"))
        class_names = ast.literal_eval(metadata.get("names", "{}"))
        self.status_changed.emit(f"模型加载完成: stride={stride}, classes={class_names}")
        print(f"DEBUG: Model loaded successfully. Stride: {stride}, Classes: {class_names}")
    except Exception as e:
        print(f"DEBUG: Error loading ONNX model: {e}")
        return f"错误: 无法加载或解析ONNX模型 '{model_path}': {e}"

    self.progress_updated.emit(0)
    providers = ['CPUExecutionProvider']
    try:
        print("DEBUG: Creating ONNX runtime session...")
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"DEBUG: ONNX session created. Input: '{input_name}', Output: '{output_name}'")
    except Exception as e:
        print(f"DEBUG: Error creating ONNX runtime session: {e}")
        return f"错误: 无法创建ONNX运行时: {e}"


    self.status_changed.emit(f"[模式3] 运行模式: CPU")
    self.status_changed.emit(f"正在对 {len(self.doc)} 个页面进行AI推理...")
    print(f"DEBUG: Starting inference for {len(self.doc)} pages.")

    TABLE_MIN_CONF = s["table_min_conf"]
    TABLE_MIN_ASPECT_RATIO, TABLE_MAX_ASPECT_RATIO, TABLE_MIN_AREA_RATIO = 0.3, 3.0, 0.01

    figures_by_page, tables_by_page = {}, {}

    num_pages = len(self.doc)
    for page_num, page in enumerate(self.doc):
        self.status_changed.emit(f"正在处理第 {page_num + 1}/{num_pages} 页...")
        print(f"DEBUG: Processing page {page_num + 1}/{num_pages}")

        try:
            pix = page.get_pixmap(dpi=100)
            img_cv = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            tensor, _, pad_info = self._preprocess_image(img_cv, (s['imgsz'], s['imgsz']), stride)
            print(f"DEBUG: Page {page_num+1} preprocessed. Tensor shape: {tensor.shape}")
            
            outputs = session.run([output_name], {input_name: tensor})[0]
            print(f"DEBUG: Page {page_num+1} inference complete. Output shape: {outputs.shape}")

            raw_detections = outputs[0]
            if len(raw_detections) == 0:
                print(f"DEBUG: Page {page_num+1} - No raw detections found.")
                continue
            
            print(f"DEBUG: Page {page_num+1} - Found {len(raw_detections)} raw detections.")

            # ... rest of the loop
            boxes, scores, class_ids = [], [], []
            for det in raw_detections:
                boxes.append(det[:4])
                scores.append(det[4])
                class_ids.append(int(det[5]))

            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)

            # Apply NMS per class
            kept_indices = []
            unique_class_ids = np.unique(class_ids)
            for class_id in unique_class_ids:
                class_mask = (class_ids == class_id)
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                
                # Use a standard IOU threshold for NMS
                iou_threshold = 0.45 
                
                indices_for_class = self._non_max_suppression(class_boxes, class_scores, iou_threshold)
                
                # Map back to original indices
                original_indices = np.where(class_mask)[0]
                kept_indices.extend(original_indices[indices_for_class])

            
            page_figs, page_tbls = [], []
            for i in kept_indices:
                det = raw_detections[i]
                confidence, class_id = det[4], int(det[5])
                cls_name = class_names.get(class_id, '').lower()
                is_figure = cls_name == 'figure'
                is_table = cls_name == 'table' and s['extract_tables']

                if not (is_figure or is_table): continue

                passes_conf = confidence > s['conf']
                if is_table and confidence > TABLE_MIN_CONF: passes_conf = True
                if not passes_conf: continue

                scaled_box = self._scale_boxes((s['imgsz'], s['imgsz']), det[:4][np.newaxis, ...], img_cv.shape[:2], pad_info)[0]

                if is_table:
                    box_w, box_h = scaled_box[2] - scaled_box[0], scaled_box[3] - scaled_box[1]
                    if box_h <= 0 or box_w <= 0: continue
                    aspect_ratio = box_w / box_h
                    area_ratio = (box_w * box_h) / (img_cv.shape[0] * img_cv.shape[1])
                    if not (TABLE_MIN_ASPECT_RATIO < aspect_ratio < TABLE_MAX_ASPECT_RATIO and area_ratio > TABLE_MIN_AREA_RATIO):
                        continue

                page_rect = page.rect
                bbox = fitz.Rect(
                    scaled_box[0] * page_rect.width / img_cv.shape[1], scaled_box[1] * page_rect.height / img_cv.shape[0],
                    scaled_box[2] * page_rect.width / img_cv.shape[1], scaled_box[3] * page_rect.height / img_cv.shape[0]
                )

                result = {'page': page_num, 'bbox': bbox, 'conf': confidence}
                if is_figure: page_figs.append(result)
                elif is_table: page_tbls.append(result)

            if page_figs: figures_by_page[page_num] = page_figs
            if page_tbls: tables_by_page[page_num] = page_tbls

            # Update progress after each page is processed
            progress = int(((page_num + 1) / num_pages) * 95)
            self.progress_updated.emit(progress)
        except Exception as e:
            print(f"DEBUG: An error occurred during page processing: {e}")
            # Optionally, re-raise or handle the error
            # For debugging, we'll just print it and continue
            self.status_changed.emit(f"错误：处理页面 {page_num + 1} 失败 - {e}")


    self.status_changed.emit("推理完成，正在整理和保存结果...")
    all_figures, all_tables = [], []
    for page_num in sorted(figures_by_page.keys()):
        all_figures.extend(self._sort_detected_objects(figures_by_page[page_num], self.doc[page_num].rect))
    if s['extract_tables']:
        for page_num in sorted(tables_by_page.keys()):
            all_tables.extend(self._sort_detected_objects(tables_by_page[page_num], self.doc[page_num].rect))

    print(f"DEBUG: Total figures found: {len(all_figures)}")
    print(f"DEBUG: Total tables found: {len(all_tables)}")

    target_numbers = self._parse_number_ranges(s["target_numbers"])
    fig_count = self._save_images(all_figures, s["output_dir"], "fig", s["dpi"], s["padding"], s["output_format"], s["optimize_m3"], target_numbers)
    tbl_count = 0
    if s['extract_tables']:
        tbl_count = self._save_images(all_tables, s["output_dir"], "table", s["dpi"], s["padding"], s["output_format"], s["optimize_m3"], None)

    self.progress_updated.emit(100)

    if not all_figures and not all_tables:
        return "AI模型未在此文档中检测到任何'figure'或'table'区域。"

    return f"成功提取了 {fig_count} 张图片和 {tbl_count} 个表格。"

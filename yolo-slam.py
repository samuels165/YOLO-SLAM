import os
import json
import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ultralytics import YOLO

# -------------------- Configuration -------------------- #
logs = False
# Set this flag to True to enable interactive annotation for adding extra boxes.
interactive_annotation = False

# Set this flag to True to load cached annotations from JSON
# Set to False to run YOLO + manual annotation and update the cache.
use_cached_annotations = False

# Room dimensions (in cm)
room_dimensions = (170, 120, 100)

# Camera Field of View (in degrees) - currently optimized for iphone 12
horizontal_fov = 90
vertical_fov = 60

# Load YOLOv8 model
model_path = "./yolov8n.pt"
model = YOLO(model_path)

# Datasets to load 
# Filenames should have file format:
# x_y_z_angleToXAxis_angleToZAxis.jpg
# angleToXAxis - angle where camera looks related to X axis
# angleToZAxis - angle where camera looks related to Z axis

images_1 = [
    "0_0_90_35_50.jpg",
    "85_0_90_90_50.jpg",
    "170_0_90_145_50.jpg",
    "170_60_90_180_50.jpg",
    "170_120_90_216_50.jpg",
    "85_120_90_270_50.jpg",
    "0_120_90_324_50.jpg",
    "0_60_90_0_50.jpg"
]

images_2 = [
    "0_0_90_35_50.jpg",
    "93_0_90_90_50.jpg",
    "170_0_90_145_50.jpg",
    "170_60_90_180_50.jpg",
    "170_120_90_216_50.jpg",
    "93_120_90_270_50.jpg",
    "0_120_90_324_50.jpg",
    "0_60_90_0_50.jpg"
]

images_3 = [
    "0_0_80_35_45.jpg",
    "85_0_80_90_45.jpg",
    "170_0_80_144_45.jpg",
    "170_60_80_180_45.jpg",
    "170_120_80_216_45.jpg",
    "85_120_80_270_45.jpg",
    "0_120_80_324_45.jpg",	
    "0_60_80_0_45.jpg",
]

images_4 = [
    "0_0_48_35_0.jpg",
    "92_0_85_90_50.jpg",
    "170_0_75_144_30.jpg",
    "170_60_78_180_25.jpg",
    "170_120_75_216_30.jpg",
    "90_120_59_270_35.jpg",
    "0_120_76_324_35.jpg",
    "0_70_66_0_40.jpg",
]
imgss = {"images_1": images_1, 
        #  "images_2": images_2, 
        #  "images_3": images_3, 
        #  "images_4": images_4
         }

dirss = ["ds_1",
        #  "ds_2",
        #  "ds_3",
        #  "ds_4"
         ]

# loop for testing purpose - can be deleted when needed
for (img_name, im), di in zip(imgss.items(), dirss):
    images = im
    dir_f = di

    # Create a debug folder to save images with drawn detections.
    debug_dir = os.path.join(dir_f, "_debug")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        logging.debug(f"Created debug directory: {debug_dir}")
    else:
        if logs: logging.debug(f"Debug directory already exists: {debug_dir}")

    # Create the cache folder if doesn't exist
    cache_folder = os.path.join(dir_f, "_cache")
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
        logging.debug(f"Created cache directory: {cache_folder}")
    else:
        logging.debug(f"Cache directory already exists: {cache_folder}")

    cache_file = os.path.join(cache_folder, "annotations.json")

    # Allowed detection classes and optional renaming.
    # ----------- This part can be removed when going to full action #
    # ----------- For now it serves for specific objects only ------ #
    allowed_classes = {"bowl", "cup", "bottle"}
    rename_classes = {"wine glass": "cup"}



    # -------------------- Utility Functions -------------------- #
    def parse_filename(filename):
        """
        Parse filename to extract camera position (x,y,z) and orientation angles.
        Expected format: "x_y_z_angleX_angleZ.jpg"
        Note: angle_z is negated for consistency.
        """
        parts = filename.split(".")[0].split("_")
        try:
            x, y, z = map(int, parts[:3])
            angle_x, angle_z = map(int, parts[3:])
        except Exception as e:
            raise ValueError(f"Filename {filename} does not match expected format: {e}")
        angle_z = -angle_z
        cam_pos = np.array([x, y, z], dtype=np.float32)
        if logs: logging.debug(f"Parsed filename '{filename}': camera_pos={cam_pos}, angle_x={angle_x}, angle_z={angle_z}")
        return cam_pos, angle_x, angle_z

    def pixel_to_angles(bbox_center, img_dims):
        img_height, img_width = img_dims
        # Calculate focal lengths based on the FoV
        f_x = (img_width / 2) / np.tan(np.radians(horizontal_fov / 2))
        f_y = (img_height / 2) / np.tan(np.radians(vertical_fov / 2))
        cx, cy = bbox_center
        # Compute angle offsets using arctan for more precise conversion.
        angle_h = np.degrees(np.arctan((cx - img_width / 2) / f_x))
        angle_v = np.degrees(np.arctan((cy - img_height / 2) / f_y))
        return angle_h, angle_v

    def bounding_box_to_line(camera_pos, angle_x, angle_z, angle_h, angle_v):
        """
        Compute the 3D direction vector (unit vector) from a camera position given its orientation
        and the additional angular offsets from the detected bounding box.
        """
        combined_angle_x = angle_x + angle_h
        combined_angle_z = angle_z + angle_v
        if logs: logging.debug(f"Combined angles: azimuth {combined_angle_x:.2f}°, elevation {combined_angle_z:.2f}°")
        
        rad_azimuth = np.radians(combined_angle_x)
        rad_elevation = np.radians(combined_angle_z)
        direction = np.array([
            np.cos(rad_elevation) * np.cos(rad_azimuth),
            np.cos(rad_elevation) * np.sin(rad_azimuth),
            np.sin(rad_elevation)
        ])
        direction = direction / np.linalg.norm(direction)
        if logs: logging.debug(f"Camera position: {camera_pos}, Direction vector: {direction}")
        return camera_pos, direction

    def normalize_vector(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def intersect_lines(line1, line2):
        p1, d1 = line1
        p2, d2 = line2
        d1 = normalize_vector(d1)
        d2 = normalize_vector(d2)
        dot = np.dot(d1, d2)
        if np.abs(1 - dot**2) < 1e-6:
            if logs: logging.debug("Lines are nearly parallel; no unique intersection.")
            return None
        r = p1 - p2
        denominator = 1 - dot**2
        t = (np.dot(d2, r) * dot - np.dot(d1, r)) / denominator
        s = (np.dot(d2, r) - np.dot(d1, r) * dot) / denominator
        closest_point_line1 = p1 + t * d1
        closest_point_line2 = p2 + s * d2
        midpoint = (closest_point_line1 + closest_point_line2) / 2.0
        midpoint[2] = max(midpoint[2], 0)
        midpoint[0] = room_dimensions[0] - midpoint[0]
        midpoint[1] = room_dimensions[1] - midpoint[1]
        if logs: logging.debug(f"Intersecting lines: midpoint={midpoint}")
        return midpoint

    def intersect_lines_weighted(detections):
        positions = []
        weights = []
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                line1 = (detections[i][0], detections[i][1])
                line2 = (detections[j][0], detections[j][1])
                intersection = intersect_lines(line1, line2)
                if intersection is not None:
                    positions.append(intersection)
                    avg_weight = (detections[i][2] + detections[j][2]) / 2
                    weights.append(avg_weight)
                    if logs: logging.debug(f"Intersection between detection {i} and {j}: {intersection} with avg_weight: {avg_weight}")
        if positions:
            positions = np.array(positions)
            centroid = np.average(positions, axis=0, weights=weights)
            if logs: logging.debug(f"Weighted centroid from {len(positions)} intersections: {centroid}")
            return centroid
        return None

    def visualize_3d_room(camera_positions, detections_by_class, object_positions, room_dimensions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([room_dimensions[0], room_dimensions[1], room_dimensions[2]])
        x, y, z = room_dimensions
        vertices = [
            [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
            [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
        ]
        faces = [
            [vertices[i] for i in [0, 1, 2, 3]],
            [vertices[i] for i in [4, 5, 6, 7]],
            [vertices[i] for i in [0, 1, 5, 4]],
            [vertices[i] for i in [1, 2, 6, 5]],
            [vertices[i] for i in [2, 3, 7, 6]],
            [vertices[i] for i in [3, 0, 4, 7]],
        ]
        for face in faces:
            ax.add_collection3d(Poly3DCollection([face], alpha=0.1, color='blue'))
        for pos in camera_positions:
            ax.scatter(*pos, color="green", s=50)
        for cls, detections in detections_by_class.items():
            if cls in object_positions:
                center = object_positions[cls]
                for detection in detections:
                    cam_pos = detection[0]
                    ax.plot(
                        [cam_pos[0], center[0]],
                        [cam_pos[1], center[1]],
                        [cam_pos[2], center[2]],
                        linestyle="--", linewidth=1, label=f"{cls} projection"
                    )
        # Annotate the object positions with both the class name and coordinates.
        for cls, position in object_positions.items():
            ax.scatter(*position, color="red", s=100)
        ax.set_xlim([0, room_dimensions[0]])
        ax.set_ylim([0, room_dimensions[1]])
        ax.set_zlim([0, room_dimensions[2]])
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
        # Create a string with calculated coordinates for all objects
        coords_text = "\n".join([
            f"{cls}: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
            for cls, position in object_positions.items()
        ])

        # Add the coordinates as a text box in the top-left corner of the figure.
        fig.text(0.02, 0.98, coords_text, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.show()

    # -------------------- Interactive Annotation UI -------------------- #
    def annotate_additional(base_img, image_name):
        """
        Opens an OpenCV window on the provided base image (which already shows YOLO detections).
        - Left-click and drag to draw additional manual boxes.
        - While drawing, the box is shown in real time.
        - Press 'd' to delete the last drawn manual box.
        - Press 'L' to label the last drawn manual box (a terminal prompt appears).
        - Press 'n' to finish annotating the current image.
        Returns a list of manual annotations in the format: {"bbox": [x1, y1, x2, y2], "label": <label>, "conf": 1.0}
        """
        boxes = []  # Stores only manual boxes.
        drawing = False
        ix, iy = -1, -1
        image_copy = base_img.copy()
        window_name = f"Annotate {image_name} (n=finish, d=delete, L=label)"
        
        def redraw_image():
            temp = base_img.copy()
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                lbl = b["label"] if b["label"] else "Unlabeled"
                label_text = f"{lbl} [{x1},{y1},{x2},{y2}]"
                cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(temp, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return temp

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, ix, iy, image_copy, boxes
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = redraw_image()
                cv2.rectangle(temp, (ix, iy), (x, y), (0, 0, 255), 2)
                cv2.imshow(window_name, temp)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                box = [ix, iy, x, y]
                boxes.append({"bbox": box, "label": ""})
                image_copy = redraw_image()
                cv2.imshow(window_name, image_copy)
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        while True:
            cv2.imshow(window_name, image_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('d'):
                if boxes:
                    boxes.pop()
                    image_copy = redraw_image()
                    cv2.imshow(window_name, image_copy)
            elif key == ord('L'):
                if boxes:
                    label = input("Enter label for the last drawn box: ").strip()
                    boxes[-1]["label"] = label
                    image_copy = redraw_image()
                    cv2.imshow(window_name, image_copy)
        cv2.destroyWindow(window_name)
        return boxes

    # -------------------- Main Processing Loop -------------------- #
    # This dictionary will hold the final (combined) annotations for each image.
    final_annotations = {}
    camera_positions = []
    detections_by_class = {cls: [] for cls in allowed_classes}

    print("Processing dataset...")

    if use_cached_annotations and os.path.exists(cache_file):
        # Load cached annotations
        with open(cache_file, "r") as f:
            final_annotations = json.load(f)
        logging.info(f"Loaded cached annotations from {cache_file}")
        
        # Process images using cached annotations (skip YOLO and manual annotation)
        for image_name in images:
            image_path = os.path.join(f"./{dir_f}", image_name)
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Unable to load image from {image_path}")
                continue
            try:
                camera_pos, angle_x, angle_z = parse_filename(image_name)
            except ValueError as ve:
                if logs: logging.error(ve)
                continue
            camera_positions.append(camera_pos)
            
            annots = final_annotations.get(image_name, [])
            # For debugging, draw annotations on the image.
            debug_img = image.copy()
            for annot in annots:
                x1, y1, x2, y2 = annot["bbox"]
                color = (0, 255, 0) if annot["source"] == "yolo" else (0, 0, 255)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                label_text = f"{annot['source']} {annot['label']} [{x1},{y1},{x2},{y2}]"
                cv2.putText(debug_img, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            debug_image_path = os.path.join(debug_dir, image_name)
            cv2.imwrite(debug_image_path, debug_img)
            logging.debug(f"Saved debug image with cached annotations: {debug_image_path}")
            
            # Process each annotation for 3D triangulation.
            for annot in annots:
                label = annot["label"].strip()
                if label not in allowed_classes:
                    logging.warning(f"Skipping annotation in {image_name} with bbox {annot['bbox']} due to invalid label '{label}'.")
                    continue
                bbox = annot["bbox"]
                bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                angle_h, angle_v = pixel_to_angles(bbox_center, image.shape[:2])
                cam_line = bounding_box_to_line(camera_pos, angle_x, angle_z, angle_h, angle_v)
                conf = annot.get("conf", 1.0)
                detections_by_class[label].append((cam_line[0], normalize_vector(cam_line[1]), conf))
    else:
        # Run YOLO detection and (optional) interactive annotation.
        for image_name in images:
            image_path = os.path.join(f"./{dir_f}", image_name)
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Unable to load image from {image_path}")
                continue
            logging.debug(f"Processing image: {image_path}")
            
            try:
                camera_pos, angle_x, angle_z = parse_filename(image_name)
            except ValueError as ve:
                logging.error(ve)
                continue
            camera_positions.append(camera_pos)
            
            # Run YOLO detection.
            results = model.predict(source=image, save=False, show=False)
            yolo_annotations = []
            base_img = image.copy()
            for result in results:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls)]
                    if cls_name in rename_classes:
                        cls_name = rename_classes[cls_name]
                    if cls_name in allowed_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        try:
                            conf = box.conf[0].item()
                        except Exception:
                            conf = 1.0
                        annot = {"bbox": [x1, y1, x2, y2], "label": cls_name, "conf": conf, "source": "yolo"}
                        yolo_annotations.append(annot)
                        cv2.rectangle(base_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f"{cls_name} {conf:.2f} [{x1},{y1},{x2},{y2}]"
                        cv2.putText(base_img, label_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If interactive annotation is enabled, let the user add extra boxes.
            additional_manual = []
            if interactive_annotation:
                print(f"\nAnnotating additional objects for image {image_name}.")
                additional_manual = annotate_additional(base_img, image_name)
                for annot in additional_manual:
                    annot["source"] = "manual"
            
            # Combine YOLO and manual annotations.
            combined_annots = yolo_annotations + additional_manual
            final_annotations[image_name] = combined_annots
            
            # For debugging, draw a final image.
            debug_img = image.copy()
            for annot in combined_annots:
                x1, y1, x2, y2 = annot["bbox"]
                color = (0, 255, 0) if annot["source"] == "yolo" else (0, 0, 255)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                label_text = f"{annot['source']} {annot['label']} [{x1},{y1},{x2},{y2}]"
                cv2.putText(debug_img, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            debug_image_path = os.path.join(debug_dir, image_name)
            cv2.imwrite(debug_image_path, debug_img)
            logging.debug(f"Saved debug image with combined annotations: {debug_image_path}")
            
            # Process each annotation to compute its 3D detection line.
            for annot in combined_annots:
                label = annot["label"].strip()
                if label not in allowed_classes:
                    logging.warning(f"Skipping annotation in {image_name} with bbox {annot['bbox']} due to invalid label '{label}'.")
                    continue
                bbox = annot["bbox"]
                bbox_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                angle_h, angle_v = pixel_to_angles(bbox_center, image.shape[:2])
                cam_line = bounding_box_to_line(camera_pos, angle_x, angle_z, angle_h, angle_v)
                conf = annot.get("conf", 1.0)
                detections_by_class[label].append((cam_line[0], normalize_vector(cam_line[1]), conf))
        
        # Save the newly computed annotations to the cache.
        with open(cache_file, "w") as f:
            json.dump(final_annotations, f, indent=4)
        logging.info(f"Saved annotations to cache file: {cache_file}")

    # Compute object positions using weighted centroids from pairwise intersections.
    object_positions = {}
    for cls, detections in detections_by_class.items():
        if len(detections) < 2:
            logging.warning(f"Not enough detections to triangulate position for {cls}")
            continue
        weighted_centroid = intersect_lines_weighted(detections)
        if weighted_centroid is not None:
            object_positions[cls] = weighted_centroid

    print("Detected objects coordinates:")
    for cls, coords in object_positions.items():
        print(f"{cls}: {coords}")

    # -------------------- Saving Results as JSON -------------------- #
    results_dir = os.path.join(dir_f, "_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.debug(f"Created results directory: {results_dir}")
    results_file = os.path.join(results_dir, "object_positions.json")
    # Convert numpy arrays to lists for JSON serialization.
    object_positions_serializable = {cls: coords.tolist() for cls, coords in object_positions.items()}
    with open(results_file, "w") as f:
        json.dump(object_positions_serializable, f, indent=4)
    logging.info(f"Saved object positions to {results_file}")

    visualize_3d_room(camera_positions, detections_by_class, object_positions, room_dimensions)

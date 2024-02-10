import cv2
import argparse

from ultralytics import YOLO
import supervision as sv




def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YoloV8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_height)


    # model = YOLO('yolo-head-detection.pt')
    model = YOLO('yolov8l.pt')

    CLASS_NAMES_DICT = model.model.names
    box_annotator = sv.BoxAnnotator()
    # mask_annotator = sv.MaskAnnotator()
    tracker = sv.ByteTrack(frame_rate=30,track_buffer=120)
    print(model.model.names)

    ret,frame = cap.read()
    cap_out = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))
    while ret:
        ids = set()
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)  
        detections = detections[(detections.class_id==0)]
        detections = tracker.update_with_detections(detections=detections)
        labels  = []
        # labels = [
        #    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        #    for _, _, confidence, class_id, tracker_id
        #    in detections
        # ]
        for _, _, confidence, class_id, tracker_id in detections:
            labels.append(f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}")
            ids.add(tracker_id)
        
        annotated_image = box_annotator.annotate(frame, detections=detections,labels=labels)
        # annotated_image = mask_annotator.annotate(scene=frame,detections=detections,opacity=0.5)
        cv2.putText(
                        annotated_image,
                        f"Count {len(ids)}",
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,0,0),
                        2
                    )
        cv2.imshow("YoloV8",annotated_image)
        cap_out.write(annotated_image)
        # ESC -> 27
        if cv2.waitKey(30) == 27:
            break
        ret,frame = cap.read()

if __name__ == "__main__":
    main()
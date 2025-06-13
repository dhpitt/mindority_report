import numpy as np
import torch
import cv2
import sys

from argparse import ArgumentParser

from torch_mediapipe.blazebase import resize_pad, denormalize_detections
from torch_mediapipe.blazeface import BlazeFace
from torch_mediapipe.blazepalm import BlazePalm
from torch_mediapipe.blazeface_landmark import BlazeFaceLandmark
from torch_mediapipe.blazehand_landmark import BlazeHandLandmark

from torch_mediapipe.visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS



def main(args):
    gpu = torch.device("mps" if torch.mps.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    if args.detect_face:
        back_detector = True

        face_detector = BlazeFace(back_model=back_detector).to(gpu)
        if back_detector:
            face_detector.load_weights("torch_mediapipe/blazefaceback.pth")
            face_detector.load_anchors("torch_mediapipe/anchors_face_back.npy")
        else:
            face_detector.load_weights("torch_mediapipe/blazeface.pth")
            face_detector.load_anchors("torch_mediapipe/anchors_face.npy")

        face_regressor = BlazeFaceLandmark().to(gpu)
        face_regressor.load_weights("torch_mediapipe/blazeface_landmark.pth")
    else:
        face_detector = face_regressor = None

    if args.detect_hands:
        palm_detector = BlazePalm().to(gpu)
        palm_detector.load_weights("torch_mediapipe/blazepalm.pth")
        palm_detector.load_anchors("torch_mediapipe/anchors_palm.npy")
        palm_detector.min_score_thresh = .75

        hand_regressor = BlazeHandLandmark().to(gpu)
        hand_regressor.load_weights("torch_mediapipe/blazehand_landmark.pth")
    else:
        palm_detector = hand_regressor = None

    WINDOW='test'
    cv2.namedWindow(WINDOW)
    if len(sys.argv) > 1:
        capture = cv2.VideoCapture(sys.argv[1])
        mirror_img = False
    else:
        capture = cv2.VideoCapture(0)
        mirror_img = True

    if capture.isOpened():
        hasFrame, frame = capture.read()
        frame_ct = 0
    else:
        hasFrame = False

    while hasFrame:
        frame_ct +=1

        if mirror_img:
            frame = np.ascontiguousarray(frame[:,::-1,::-1])
        else:
            frame = np.ascontiguousarray(frame[:,:,::-1])

        img1, img2, scale, pad = resize_pad(frame)
        
        if args.detect_face:
            if back_detector:
                normalized_face_detections = face_detector.predict_on_image(img1)
            else:
                normalized_face_detections = face_detector.predict_on_image(img2)
            face_detections = denormalize_detections(normalized_face_detections, scale, pad)

        else:
            face_detections = normalized_face_detections = None
        
        if args.detect_hands:
            normalized_palm_detections = palm_detector.predict_on_image(img1)

            palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

        
        if args.detect_face:
            xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
            img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
            flags, normalized_landmarks = face_regressor(img.to(gpu))
            landmarks = face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)
        else:
            landmarks = None


        if args.detect_hands:
            xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
            img, affine2, box2 = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
            flags2, handed2, normalized_landmarks2 = hand_regressor(img.to(gpu))
            landmarks2 = hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine2)
            print(f"{box2.shape=}")
            print(f"{landmarks2.shape=}")
            print(f"{[x.shape for x in flags2]=}")
            print(f"{flags2=}")
        else:
            landmarks2 = None

        if args.draw and landmarks is not None:
            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                if flag>.5:
                    draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)
            draw_roi(frame, box)
        
        if args.draw and landmarks2 is not None:
            for i in range(len(flags2)):
                landmark, flag = landmarks2[i], flags2[i]
                if flag>.5:
                    draw_landmarks(frame, landmark[:,:2], HAND_CONNECTIONS, size=2)
            draw_roi(frame, box2)
        if args.draw:
            if face_detections is not None:
                draw_detections(frame, face_detections)
            if palm_detections is not None:
                draw_detections(frame, palm_detections)

        cv2.imshow(WINDOW, frame[:,:,::-1])
        # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

        hasFrame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


# create argparser
parser = ArgumentParser()
parser.add_argument("--detect_face", action="store_true", default=False)
parser.add_argument("--detect_hands", action="store_true", default=True)
parser.add_argument("--draw", action="store_true", default=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

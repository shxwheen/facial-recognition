import cv2
import numpy as np
import face_recognition
import argparse


def compare_images(image1_path, image2_path, show_images=True):
    # Load and process images
    img1 = face_recognition.load_image_file(image1_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = face_recognition.load_image_file(image2_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Detect faces
    faceLocation1 = face_recognition.face_locations(img1)
    faceLocation2 = face_recognition.face_locations(img2)
    if not faceLocation1 or not faceLocation2:
        print("Could not detect a face in one or both images.")
        return
    faceLocation1 = faceLocation1[0]
    faceLocation2 = faceLocation2[0]
    encode1 = face_recognition.face_encodings(img1)[0]
    encode2 = face_recognition.face_encodings(img2)[0]

    # Draw rectangles
    cv2.rectangle(img1, (faceLocation1[3], faceLocation1[0]), (faceLocation1[1], faceLocation1[2]), (255, 0, 255), 2)
    cv2.rectangle(img2, (faceLocation2[3], faceLocation2[0]), (faceLocation2[1], faceLocation2[2]), (255, 0, 255), 2)

    # Compare
    results = face_recognition.compare_faces([encode1], encode2)
    faceDistance = face_recognition.face_distance([encode1], encode2)
    print(f"Match: {results[0]}, Face Distance: {faceDistance[0]}")
    cv2.putText(img2, f'{results} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if show_images:
        cv2.imshow("Image 1", img1)
        cv2.imshow("Image 2", img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two images for face similarity.")
    parser.add_argument("image1", help="Path to the first image.")
    parser.add_argument("image2", help="Path to the second image.")
    parser.add_argument("--no-show", action="store_true", help="Do not display images.")
    args = parser.parse_args()
    compare_images(args.image1, args.image2, show_images=not args.no_show)


import time
import torch
import cv2
import os
import csv
import re

from ultralytics import YOLO
from easyocr import Reader
from datetime import datetime

CONFIDENCE_THRESHOLD = 0.4
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    start = time.time()
    detections = model.predict(image)[0].boxes.data

    if detections.shape != torch.Size([0, 6]):
        boxes = []
        confidences = []

        for detection in detections:
            confidence = detection[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue
            boxes.append(detection[:4])
            confidences.append(detection[4])

        print(f"{len(boxes)} Number plate(s) have been detected.")
        number_plate_list = []

        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "Number Plate: {:.2f}%".format(confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        end = time.time()
        print(f"Time to detect the number plates: {(end - start) * 1000:.0f} milliseconds")
        return number_plate_list, image
    else:
        print("Tidak terdapat plat nomor terdeteksi.")
        return [], image

def recognize_number_plates(image_or_path, reader,
                            number_plate_list, write_to_csv=False):

    start = time.time()
    # if the image is a path, load the image; otherwise, use the image
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str)\
                                        else image_or_path

    for i, box in enumerate(number_plate_list):
        # crop the number plate region
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        # detect the text from the license plate using the EasyOCR reader
        detection = reader.readtext(np_image, paragraph=True)

        if len(detection) == 0:
            # if no text is detected, set the `text` variable to an empty string
            text = ""
        else:
            # set the `text` variable to the detected text
            text = str(detection[0][1])

        # update the `number_plate_list` list, adding the detected text
        number_plate_list[i].append(text)

    if write_to_csv:
        # open the CSV file
        csv_file = open("number_plates.csv", "w")
        # create a writer object
        csv_writer = csv.writer(csv_file)
        # write the header
        csv_writer.writerow(["image_path", "box", "text"])

        # loop over the `number_plate_list` list
        for box, text in number_plate_list:
            # write the image path, bounding box coordinates,
            # and detected text to the CSV file
            csv_writer.writerow([image_or_path, box, text])
        # close the CSV file
        csv_file.close()

    end = time.time()
    # show the time it took to recognize the number plates
    print(f"Time to recognize the number plates: {(end - start) * 1000:.0f} milliseconds")

    return number_plate_list
    
def extract_tax_info(text):
    """
    Ekstrak informasi bulan dan tahun pajak dari teks plat nomor
    Format yang diharapkan: dua digit terakhir tahun (contoh: 23) 
    dan bulan (contoh: 07) yang berdekatan
    """
    # Cari pola 2-4 digit berurutan (untuk menemukan bulan dan tahun)
    matches = re.findall(r'\d{2,4}', text)
    
    if len(matches) >= 2:
        # Ambil 2 digit terakhir dari match terakhir sebagai tahun
        tax_year = matches[-1][-2:]
        
        # Ambil 2 digit sebelum tahun sebagai bulan
        if len(matches[-1]) >= 4:
            tax_month = matches[-1][:2]
        elif len(matches) >= 2:
            tax_month = matches[-2][-2:]
        else:
            return None, None
        
        return tax_month, tax_year
    return None, None

def validate_tax(tax_month, tax_year):
    """Validasi status pajak - versi sederhana"""
    if not tax_month or not tax_year:
        return "Tidak Diketahui", "Informasi tidak lengkap"
    
    try:
        # Konversi ke integer
        month = int(tax_month)
        year = int(tax_year) + 2000  # Asumsi tahun 2000+
        
        # Validasi bulan
        if month < 1 or month > 12:
            return "Invalid", "Bulan tidak valid"
        
        # Tanggal akhir masa pajak
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        
        tax_end_date = datetime(next_year, next_month, 1)
        current_date = datetime.now()
        
        # Tentukan status
        status = "AKTIF" if current_date <= tax_end_date else "KADALUARSA"
        
        nama_bulan = [
            "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Oktober", "November", "Desember"
        ][month-1]
        
        masa_berlaku = f"{nama_bulan} {year}"
        return status, masa_berlaku
    
    except:
        return "Error", "Terjadi kesalahan"

# if this script is executed directly, run the following code
if __name__ == "__main__":

    # load the model from the local directory
    model = YOLO("best.pt")
    # initialize the EasyOCR reader
    reader = Reader(['en'], gpu=True)

    # path to an image or a video file
    file_path = "gambar_google_10_1.jpg"
    # Extract the file name and the file extension from the file path
    _, file_extension = os.path.splitext(file_path)

    # Check the file extension
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing the image...")

        image = cv2.imread(file_path)
        number_plate_list = detect_number_plates(image, model,
                                                display=True)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

        # if there are any number plates detected, recognize them
        if number_plate_list != []:
            number_plate_list = recognize_number_plates(file_path, reader,
                                                        number_plate_list,
                                                        write_to_csv=True)

            for box, text in number_plate_list:
                cv2.putText(image, text, (box[0], box[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            cv2.imshow('Image', image)
            cv2.waitKey(0)

    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        print("Processing the video...")

        video_cap = cv2.VideoCapture(file_path)

        # grab the width and the height of the video stream
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        # initialize the FourCC and a video writer object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps,
                                (frame_width, frame_height))

        # loop over the frames
        while True:
            # starter time to computer the fps
            start = time.time()
            success, frame = video_cap.read()

            # if there is no more frame to show, break the loop
            if not success:
                print("There are no more frames to process."
                        " Exiting the script...")
                break

            number_plate_list = detect_number_plates(frame, model)

            if number_plate_list != []:
                number_plate_list = recognize_number_plates(frame, reader,
                                                        number_plate_list)

                for box, text in number_plate_list:
                    cv2.putText(frame, text, (box[0], box[3] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR, 2)

            # end time to compute the fps
            end = time.time()
            # calculate the frame per second and draw it on the frame
            fps = f"FPS: {1 / (end - start):.2f}"
            cv2.putText(frame, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            # show the output frame
            cv2.imshow("Output", frame)
            # write the frame to disk
            writer.write(frame)
            # if the 'q' key is pressed, break the loop
            if cv2.waitKey(10) == ord("q"):
                break

        # release the video capture, video writer, and close all windows
        video_cap.release()
        writer.release()
        cv2.destroyAllWindows()

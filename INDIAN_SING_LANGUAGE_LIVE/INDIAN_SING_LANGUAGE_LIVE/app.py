from flask import Flask, render_template, request, url_for, send_from_directory, Response
from collections import Counter
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator
import cv2
import datetime
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import google.generativeai as genai
from google.api_core import exceptions as api_exceptions

app = Flask(__name__)

# Folder configurations
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
REPORT_FOLDER = 'reports'
OUTPUT_FOLDER = 'static/outputs'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, REPORT_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load YOLO model for Indian Sign Language
model_path = "best.pt"  # Hypothetical model file
model = YOLO(model_path)

# Configure Gemini client
genai.configure(api_key="AIzaSyAjY_6pWDd9JZc7xPblghxbkt490EiRD7w")

# Simple in-memory circuit breaker for Gemini calls
GEMINI_DISABLED_UNTIL = 0
GEMINI_COOLDOWN_SECONDS = 300  # cooldown when quota errors occur (seconds)

# Email configurations
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = 'your-email@gmail.com'
EMAIL_PASSWORD = 'your-email-password'
EMAIL_TO = 'prosdgunal@gmail.com'

def send_email_alert(sign_list, report_filename, report_path):
    """Send an email alert with detection results and report attachment."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = 'Indian Sign Language Detection Alert'

        body = f"""
        Indian Sign Language Activity Detected!

        Detected Signs: {', '.join(sign_list)}

        A detailed report is attached for your reference.
        """
        msg.attach(MIMEText(body, 'plain'))

        with open(report_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {report_filename}'
        )
        msg.attach(part)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_image(results, model, image_path):
    """Process image detection results and save annotated image."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_image.jpg')
    image = cv2.imread(image_path)
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
        img = annotator.result()
        cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")
    return 'outputs/output_image.jpg'

def process_video(video_path, model):
    """Process video detection results and save annotated video with reduced FPS."""
    output_path = os.path.join(OUTPUT_FOLDER, 'output_video.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_fps = 5  # Set reduced FPS for slower video processing

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    sign_list = []
    frame_count = 0
    frame_interval = max(1, int(input_fps / output_fps))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        results = model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                annotator.box_label(b, model.names[int(c)])
                sign_list.append(model.names[int(c)])

        annotated_frame = annotator.result()
        out.write(annotated_frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")
    return 'outputs/output_video.mp4', list(set(sign_list))

def analyze_with_gemini(image_path, max_retries=3, initial_backoff=1.0):
    """Analyze image using Gemini-2.0-flash model with retry/backoff and graceful fallback.

    This function implements:
    - a simple in-memory cooldown/circuit-breaker when quota is exhausted,
    - exponential backoff retries for transient errors,
    - a safe fallback string when analysis cannot be obtained.
    """
    global GEMINI_DISABLED_UNTIL

    # If we previously detected quota exhaustion, skip calling Gemini until cooldown expires
    if time.time() < GEMINI_DISABLED_UNTIL:
        return "Gemini analysis temporarily disabled due to quota/exhaustion."

    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    prompt = f"Analyze the following image for Indian Sign Language gestures and context: {image_path}"

    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            # Some SDKs return a .text field, others may return object/string
            if hasattr(response, 'text'):
                return response.text
            return str(response)

        except api_exceptions.ResourceExhausted as e:
            # Quota exhausted — set cooldown and return a friendly message
            print(f"Gemini quota exhausted (attempt {attempt}): {e}")
            GEMINI_DISABLED_UNTIL = time.time() + GEMINI_COOLDOWN_SECONDS
            return "Gemini analysis unavailable due to API quota limits."

        except Exception as e:
            # For other errors, retry with exponential backoff up to max_retries
            print(f"Gemini API error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2
                continue
            # Final fallback
            return f"Gemini analysis failed: {e}"

def run_object_detection(file_path, is_video=False):
    """Run object detection on image or video."""
    if is_video:
        output_path, sign_list = process_video(file_path, model)
        print("Detected signs:", sign_list)
    else:
        results = model.predict(file_path)
        sign_counts = Counter(model.names[int(c)] for r in results for c in r.boxes.cls)
        sign_list = list(sign_counts.keys())
        print("Detected signs:", sign_list)
        output_path = process_image(results, model, file_path)

    gemini_analysis = analyze_with_gemini(file_path)
    print("Gemini Analysis:", gemini_analysis)

    return sign_list, output_path, gemini_analysis

def gen_frames():
    """Generate frames for live camera feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results[0].names[int(box.cls[0])]
            confidence = round(box.conf[0].item(), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def landing():
    """Render landing page."""
    return render_template('landing.html')

@app.route('/index')
def upload():
    """Render upload page."""
    return render_template('index.html')

@app.route('/live')
def live():
    """Render live camera page."""
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    """Stream live camera feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_file', methods=['POST'])
def process_file():
    """Process uploaded file and run detection."""
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    allowed_video_extensions = {'mp4', 'avi', 'mov'}
    filename = secure_filename(file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    if file_ext not in allowed_image_extensions and file_ext not in allowed_video_extensions:
        return render_template('result.html', error="Invalid file type")

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    is_video = file_ext in allowed_video_extensions
    sign_list, output_path, gemini_analysis = run_object_detection(file_path, is_video)
    if not output_path:
        return render_template('result.html', error="Error processing video")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{os.path.splitext(filename)[0]}_{timestamp}_report.txt"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    # Write report using UTF-8 to avoid UnicodeEncodeError on Windows (cp1252)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(
            f"Indian Sign Language Detection Report\n\nDetected Signs: {', '.join(sign_list)}\n\nTimestamp: {timestamp}\n\nGemini Analysis: {gemini_analysis}"
        )

    send_email_alert(sign_list, report_filename, report_path)

    return render_template(
        'result.html',
        filename=filename,
        sign_list=sign_list,
        output_path=output_path,
        is_video=is_video,
        report_filename=report_filename,
        gemini_analysis=gemini_analysis
    )

@app.route('/static/outputs/<filename>')
def outputs(filename):
    """Serve output files."""
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download report file."""
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
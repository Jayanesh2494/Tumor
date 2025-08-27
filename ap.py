from flask import Flask, request, jsonify, session, send_file, send_from_directory, render_template, redirect, url_for
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import bcrypt
import os
import uuid
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
import json
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
import shutil
import subprocess
from dotenv import load_dotenv
load_dotenv()

# Import your custom model utilities
from model_backend import HybridCNNViT, load_checkpoint_into_model, run_gradcampp

from llm import (
    CONFIG,
    load_ensemble_model,
    get_image_transform,
    GradCAMPlusPlus,
    build_gradcam_overlay_pil,
    generate_ai_description,
    clean_ai_text,
    create_pdf
)

app = Flask(__name__, static_folder="assets")
app.secret_key = 'SECRET_KEY'  # Change this in production
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = "reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# MODEL_PATH = '/Users/keerthevasan/Documents/Study/Tumor_frontend/final.h5'  # Path to your trained model
# IMG_SIZE = (299, 299)  # Adjust based on your model's input shape
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'neurodetect',
    'user': 'neuro_user',
    'password': 'root'
}

# Create upload directory if it doesn't exist
'''os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load ML model
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✅ ML Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
'''
# Tumor classes - adjust based on your model
TUMOR_CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    if not conn:
        print("❌ Cannot connect to database")
        return
    
    try:
        cursor = conn.cursor()
               # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name VARCHAR(100),
                email VARCHAR(100),
                role VARCHAR(20) DEFAULT 'doctor',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                patient_id VARCHAR(20) UNIQUE NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                age INTEGER NOT NULL,
                gender VARCHAR(10) NOT NULL,
                scan_date DATE NOT NULL,
                symptoms TEXT,
                created_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_email VARCHAR(255)
            );
        """)
        
        # Create scans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id SERIAL PRIMARY KEY,
                patient_id INTEGER REFERENCES patients(id),
                scan_image_path TEXT NOT NULL,
                gradcam_image_path TEXT,
                prediction_result VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                tumor_type VARCHAR(50),
                processing_time FLOAT,
                model_version VARCHAR(20) DEFAULT 'CNN-v3.1',
                image_quality VARCHAR(20),
                scan_type VARCHAR(20),
                ai_explanation TEXT,
                doctor_notes TEXT,
                status VARCHAR(20) DEFAULT 'pending',
                analyzed_by INTEGER REFERENCES users(id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create system_stats table for dashboard
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id SERIAL PRIMARY KEY,
                total_scans INTEGER DEFAULT 0,
                accuracy_rate FLOAT DEFAULT 0.0,
                avg_processing_time FLOAT DEFAULT 0.0,
                early_detections INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
 
        # Insert default admin user if not exists
        cursor.execute("SELECT * FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            admin_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, role) 
                VALUES (%s, %s, %s, %s)
            """, ('admin', admin_password.decode('utf-8'), 'System Admin', 'admin'))
        
        # Insert demo doctor user
        cursor.execute("SELECT * FROM users WHERE username = 'doctor'")
        if not cursor.fetchone():
            doctor_password = bcrypt.hashpw('doctor123'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, role) 
                VALUES (%s, %s, %s, %s)
            """, ('doctor', doctor_password.decode('utf-8'), 'Dr. Smith', 'doctor'))
        
        # Initialize system stats if not exists
        cursor.execute("SELECT * FROM system_stats")
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO system_stats (total_scans, accuracy_rate, avg_processing_time, early_detections)
                VALUES (%s, %s, %s, %s)
            """, (0, 98.7, 1, 0))
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        conn.rollback()
        cursor.close()
        conn.close()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- UTILS ---------------- #


'''def generate_gradcam(image_path, prediction_class):
    """Generate Grad-CAM visualization (simplified version)"""
    try:
        # For now, create a simple heatmap overlay
        # In production, implement actual Grad-CAM
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Create random heatmap for demo (replace with actual Grad-CAM)
        heatmap = np.random.rand(height, width) * 0.3
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
        # Save Grad-CAM image
        gradcam_filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        gradcam_path = os.path.join(UPLOAD_FOLDER, gradcam_filename)
        cv2.imwrite(gradcam_path, overlay)
        
        return gradcam_path
    except Exception as e:
        print(f"Grad-CAM generation error: {e}")
        return None'''

def get_tumor_explanation(tumor_type):
    """Get AI explanation for tumor type"""
    explanations = {
        'glioma': "Gliomas are tumors that arise from glial cells in the brain and spinal cord. The AI detected characteristic irregular boundaries and heterogeneous signal intensities typical of glial cell proliferation, indicating potential malignant transformation requiring immediate medical attention.",
        'meningioma': "Meningiomas originate from the meninges surrounding the brain and are typically benign slow-growing tumors. The AI identified the characteristic well-defined borders and homogeneous enhancement pattern, suggesting a tumor arising from the protective membrane layers of the brain.",
        'pituitary': "Pituitary adenomas are typically benign tumors of the pituitary gland that can affect hormone production. The AI detected an abnormal mass in the sella turcica region with characteristic enhancement patterns, potentially disrupting normal pituitary hormone regulation.",
        'normal': "The brain scan appears normal with no detectable tumor masses or suspicious lesions. The AI analysis shows healthy brain tissue architecture with normal signal intensities, symmetrical structures, and no evidence of abnormal cell growth or space-occupying lesions."
    }
    return explanations.get(tumor_type, "Analysis completed. Please consult with a medical professional for detailed interpretation.")

# Routes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/sign-in')
        return f(*args, **kwargs)
    return decorated_function

@app.route("/sign-in")
def sign_in_page():
    return render_template("pages/sign-in.html")

@app.route("/sign-up")
def sign_up_page():
    return render_template("pages/sign-up.html")

@app.route("/dashboard")
@login_required
def dashboard_page():
    return render_template("pages/dashboard.html")

@app.route("/profile")
@login_required
def profile_page():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Fetch user info using session['user_id']
    cursor.execute("""
        SELECT username, full_name, email, role, created_at
        FROM users WHERE id = %s
    """, (session['user_id'],))
    
    user = cursor.fetchone()
    
    conn.close()
    return render_template("pages/profile.html", user=user)


@app.route("/tables")
@login_required
def tables_page():
    return render_template("pages/tables.html")

@app.route("/predict")
@login_required
def predict_page():
    return render_template("pages/predict.html")

@app.route("/visualization")
@login_required
def visualization_page():
    return render_template("pages/visualization.html")

@app.route("/services")
def services_page():
    return render_template("pages/services.html")

@app.route("/about")
def about_us_page():
    return render_template("pages/about.html")

@app.route("/")
def serve_main():
    return redirect(url_for("sign_in_page"))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/reports/<filename>')
def report_file(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(os.path.join(app.root_path, 'results'), filename)

@app.route('/api/register', methods=['POST'])
def register_user():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')
    full_name = data.get('full_name', '')
    email = data.get('email', '')

    if not username or not password or not email:
        return jsonify({'success': False, 'message': 'Username, password, and email are required.'}), 400

    # Hash password
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            return jsonify({'success': False, 'message': 'Username already exists.'}), 409

        # Insert new user
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email)
            VALUES (%s, %s, %s, %s)
            RETURNING id, username, full_name, email, created_at;
        """, (username, password_hash, full_name, email))
        user = cursor.fetchone()
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({'success': True, 'message': 'User registered successfully.', 'user': user}), 201

    except Exception as e:
        print("Registration error:", e)
        return jsonify({'success': False, 'message': 'An error occurred during registration.'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']
            
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True, 
                'message': 'Login successful',
                'user': {
                    'id': user['id'],
                    'username': user['username'],
                    'full_name': user['full_name'],
                    'role': user['role']
                }
            })
        else:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/sign-in') 

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Fetch dashboard statistics dynamically from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Total scans
        cursor.execute("SELECT COUNT(*) as total_scans FROM scans")
        total_scans = cursor.fetchone()['total_scans'] or 0

        # 2. Tumor detected vs no tumor
        cursor.execute("SELECT COUNT(*) as tumor_detected FROM scans WHERE tumor_type != 'notumor'")
        tumor_detected = cursor.fetchone()['tumor_detected'] or 0

        cursor.execute("SELECT COUNT(*) as no_tumor FROM scans WHERE tumor_type = 'notumor'")
        no_tumor = cursor.fetchone()['no_tumor'] or 0

        # 3. Early detections (tumor_type != 'normal')
        cursor.execute("SELECT COUNT(*) as early_detections FROM scans WHERE tumor_type != 'notumor'")
        early_detections = cursor.fetchone()['early_detections'] or 0

        # 4. Average processing time
        cursor.execute("SELECT AVG(processing_time) as avg_time FROM scans")
        avg_time = cursor.fetchone()['avg_time'] or 0

        # 5. Breakdown by tumor types
        tumor_breakdown = {}
        for t in TUMOR_CLASSES:
            cursor.execute("SELECT COUNT(*) as count FROM scans WHERE tumor_type = %s", (t,))
            tumor_breakdown[t] = cursor.fetchone()['count'] or 0

        cursor.close()
        conn.close()

        stats = {
            "total_scans": total_scans,
            "tumor_detected": tumor_detected,
            "no_tumor": no_tumor,
            "early_detections": early_detections,
            "avg_processing_time": round(avg_time, 2),
            "accuracy_rate": 98.7,  # placeholder, unless you want to compute dynamically
            "tumor_breakdown": tumor_breakdown
        }

        return jsonify({"success": True, "stats": stats})

    except Exception as e:
        return jsonify({"success": False, "message": f"Server error: {str(e)}"}), 500

@app.route('/api/dashboard/tumor-breakdown', methods=['GET'])
def tumor_breakdown():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Example query: count per category
    cursor.execute("""
        SELECT tumor_type, COUNT(*) as total 
        FROM scans 
        GROUP BY tumor_type
    """)
    results = cursor.fetchall()

    # Calculate percentages
    total_scans = sum(r['total'] for r in results)
    for r in results:
        r['percentage'] = round((r['total'] / total_scans) * 100, 2)

    return jsonify(success=True, data=results)

@app.route("/api/recent-predictions", methods=["GET"])
def get_recent_predictions():
    """Fetch recent predictions timeline"""
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "message": "DB connection failed"}), 500
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                p.patient_id,
                s.prediction_result,
                s.tumor_type,
                s.created_at AS timestamp
            FROM scans s
            JOIN patients p ON s.patient_id = p.id
            ORDER BY s.created_at DESC
            LIMIT 3;
        """)
        rows = cursor.fetchall()
        
        data = []
        for row in rows:
            # Format output
            prediction_text = (
                f"Tumor detected ({row['tumor_type'].capitalize()})"
                if row["tumor_type"].lower() != "notumor"
                else "No Tumor"
            )
            data.append({
                "patient_id": row["patient_id"],
                "prediction_result": prediction_text,
                "timestamp": row["timestamp"].isoformat()
            })
        
        return jsonify({"success": True, "data": data})
    
    except Exception as e:
        print(f"DB Query error: {e}")
        return jsonify({"success": False, "message": "Query failed"}), 500
    finally:
        conn.close()


@app.route('/api/patient/create', methods=['POST'])
def create_patient():
    """Create new patient record"""
    try:
        data = request.get_json()
        
        required_fields = ['patient_id', 'full_name', 'age', 'gender', 'scan_date','user_email']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'message': f'{field} is required'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO patients (patient_id, full_name, age, gender, scan_date, symptoms, user_email, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            data['patient_id'],
            data['full_name'],
            data['age'],
            data['gender'],
            data['scan_date'],
            data.get('symptoms', ''),
            data['user_email'],
            session.get('user_id')
        ))
        
        patient_db_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Patient created successfully',
            'patient_id': patient_db_id
        })
        
    except psycopg2.IntegrityError:
        return jsonify({'success': False, 'message': 'Patient ID already exists'}), 409
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
MODEL_PATH = "hybrid_cnn_vit_best.pt"

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

# Rebuild model
model = HybridCNNViT(num_classes=len(CLASSES)).to(DEVICE)

# Load only the weights from ckpt["model"]
model.load_state_dict(ckpt["model"])
model.eval()


@app.route('/api/scan/upload', methods=['POST'])
def upload_and_analyze_scan():
    """Upload brain scan and analyze with AI"""
    try:
        file = request.files['file']
        patient_id = request.form.get('patient_id')

        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        if not patient_id:
            return jsonify({'success': False, 'message': 'Patient ID required'}), 400
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type'}), 400

        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        results_folder = os.path.join(app.root_path, 'results')
        os.makedirs(results_folder, exist_ok=True)

        # ---- Run AI analysis (Hybrid model + GradCAM) ---- #
        start_time = datetime.now()
        results = run_gradcampp(model, file_path, mask_path=None, threshold=0.3)
        processing_time = (datetime.now() - start_time).total_seconds()

        pred_class = results['pred_class']
        tumor_type = results['pred_class']

        confidence = results['confidence']
        gradcam_filename = results['gradcam_file']
        gradcam_path = os.path.join(results_folder, gradcam_filename)

        # Move overlay into results folder
        if os.path.exists(gradcam_filename):
            shutil.move(gradcam_filename, gradcam_path)

        # ---- Database operations ---- #
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT id, full_name, age FROM patients WHERE patient_id = %s", (patient_id,))
        patient = cursor.fetchone()

        if not patient:
            return jsonify({'success': False, 'message': 'Patient not found'}), 404

        # AI explanation
        explanation = get_tumor_explanation(pred_class)

        # Determine result status
        result_status = "No Tumor Detected" if pred_class == 'notumor' else f"{pred_class.capitalize()} Detected"


        
        
        # Save scan results to database
        cursor.execute("""
            INSERT INTO scans (
                patient_id, scan_image_path, gradcam_image_path, 
                prediction_result, confidence, tumor_type,
                processing_time, image_quality, scan_type, ai_explanation,
                analyzed_by
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            patient['id'], file_path, gradcam_path,
            result_status, confidence, tumor_type,
            processing_time, 'Excellent', 'MRI T1', explanation,
            session.get('user_id')
        ))
        
        
        
        # Update system stats
        cursor.execute("""
            UPDATE system_stats SET 
                total_scans = total_scans + 1,
                avg_processing_time = ROUND(((avg_processing_time + %s) / 2)::numeric, 2)::double precision,
                last_updated = CURRENT_TIMESTAMP
        """, (processing_time,))
        conn.commit()

        

        # After you calculate:
        # tumor_type, confidence, explanation
        try:
            # 1️⃣ Prepare report data
            report_data = {
                "patient_id": patient_id,
                "patient_name": patient.get('full_name', 'Unknown'),
                "age": patient.get('age', 'Unknown'),
                "scan_date": str(patient.get('scan_date', datetime.now().date())),
                "tumor_type": tumor_type,
                "confidence": round(confidence * 100, 1)
            }

            # 2️⃣ Generate AI text description
            ai_text = generate_ai_description(report_data)

            # 3️⃣ Construct PDF filename and path
            pdf_filename = f"report_scan_{patient['id']}.pdf"
            pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)

            # 4️⃣ Collect images for PDF
            images_for_pdf = [(file_path, "Original MRI Scan")]

            # Append Grad-CAM overlay if it exists
            
            if os.path.exists(gradcam_path):
                images_for_pdf.append((gradcam_path, "Grad-CAM++ Heatmap (Model focus)"))

            # 5️⃣ Generate PDF
            create_pdf(pdf_path, report_data, ai_text, images_for_pdf)

            # 6️⃣ Construct URL for frontend
            report_url = f'/reports/{pdf_filename}' if os.path.exists(pdf_path) else None

        except Exception as report_error:
            print("Report generation failed:", report_error)
            report_url = None


        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'result': {
                'status': result_status,
                'confidence': round(confidence, 3),
                'tumor_type': tumor_type,
                'processing_time': round(processing_time, 3),
                'explanation': explanation,
                'gradcam_overlay_path': f'/results/{gradcam_filename}' ,
                'report_path': report_url,
                'original_mri': f'/uploads/{filename}'
            }
        })
        
    except Exception as e:
        print("Exception occurred:\n", traceback.format_exc())
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500
    
    


@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all patient reports"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                p.patient_id,
                p.full_name,
                p.scan_date::text as scan_date,
                s.prediction_result,
                s.confidence,
                s.tumor_type,
                s.status,
                s.id as scan_id,
                s.created_at
            FROM patients p
            JOIN scans s ON p.id = s.patient_id
            ORDER BY s.created_at DESC
        """)
        
        reports = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Format reports for frontend
        formatted_reports = []
        for report in reports:
            formatted_reports.append({
                'patient_id': report['patient_id'],
                'patient_name': report['full_name'],
                'scan_date': report['scan_date'] or report['created_at'].strftime('%Y-%m-%d'),
                'result': report['prediction_result'],
                'confidence': f"{report['confidence']:.0f}%",
                'tumor_type': report['tumor_type'],
                'status': report['status'],
                'scan_id': report['scan_id']
            })
        
        return jsonify({'success': True, 'reports': formatted_reports})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500
    


@app.route('/api/patients/<patient_uid>', methods=['DELETE'])
def delete_patient(patient_uid):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # find the internal numeric id for this external patient_id
        cur.execute("SELECT id FROM patients WHERE patient_id = %s", (patient_uid,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({'success': False, 'message': 'Patient not found'}), 404

        internal_id = row[0]

        # delete scans first (FK constraint)
        cur.execute("DELETE FROM scans WHERE patient_id = %s", (internal_id,))
        scans_deleted = cur.rowcount

        # delete patient
        cur.execute("DELETE FROM patients WHERE id = %s", (internal_id,))
        patient_deleted = cur.rowcount

        conn.commit()
        cur.close(); conn.close()

        return jsonify({
            'success': True,
            'message': 'Deleted',
            'deleted_scans': scans_deleted,
            'deleted_patient': patient_deleted
        })
    except Exception as e:
        # make sure we close connection on error
        try:
            conn.rollback()
            cur.close(); conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': str(e)}), 500




@app.route('/api/scan/<int:scan_id>/status', methods=['POST'])
def update_scan_status(scan_id):
    try:
        new_status = request.json.get('status')  # expected 'reviewing' or 'completed'
        if new_status not in ['reviewing', 'completed']:
            return jsonify({'success': False, 'message': 'Invalid status'}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE scans SET status = %s WHERE id = %s RETURNING id", (new_status, scan_id))
        updated = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        if updated:
            return jsonify({'success': True, 'message': f'Status updated to {new_status}'})
        else:
            return jsonify({'success': False, 'message': 'Scan not found'}), 404

    except Exception as e:
        try:
            conn.rollback()
            cur.close(); conn.close()
        except:
            pass
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_file():
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        if not file.filename.endswith(".nii"):
            return jsonify({"error": "Only .nii files allowed"}), 400

        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Run model.py with uploaded file
        process = subprocess.Popen(
            ["python3", "model.py", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return jsonify({
                "error": "Model script failed",
                "stderr": stderr.decode("utf-8")
            }), 500

        return jsonify({
            "message": "Analysis started",
            "output": stdout.decode("utf-8")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




# Initialize database on startup
init_database()

if __name__ == '__main__':
    print("🧠 NeuroDetect Backend Server Starting...")
    print("📊 Dashboard: http://localhost:5001")
    print("🔐 Default Login: admin/admin123 or doctor/doctor123")
    app.run(debug=True, host='0.0.0.0', port=5001)
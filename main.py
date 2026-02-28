from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()

# ---------------------------------------------------------
# 1. ตั้งค่า CORS (สำคัญมาก! เพื่อให้มือถือ/เว็บอื่นเรียกใช้ได้)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุกเว็บ (แก้ปัญหา CORS Block)
    allow_credentials=True,
    allow_methods=["*"],  # อนุญาตทุกคำสั่ง (GET, POST, etc.)
    allow_headers=["*"],  # อนุญาตทุก Header (รวมถึง ngrok-skip-browser-warning)
)

# ---------------------------------------------------------
# 2. โหลดสมอง AI
# ---------------------------------------------------------
# ลองโหลด best.pt ก่อน (โมเดลไก่) ถ้าไม่มีจะใช้ yolov8n.pt แทน
try:
    model = YOLO("best.pt")
    print("✅ โหลดโมเดลไก่ (best.pt) เรียบร้อย!")
except Exception as e:
    print(f"⚠️ ไม่เจอไฟล์ best.pt ({e}) -> กำลังใช้โมเดลมาตรฐาน yolov8n.pt แทน")
    model = YOLO("yolov8n.pt")

@app.get("/")
def read_root():
    return {"status": "AI Server is online and ready!"}

@app.post("/count-chickens")
async def count_chickens(file: UploadFile = File(...)):
    try:
        # 1. อ่านรูปภาพ
        contents = await file.read()
        
        # 🛑 ป้องกันภาพ Error จากกล้องมือถือ
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 2. ให้ AI ทำนาย (conf=0.25 คือความมั่นใจ 25% ขึ้นไปถึงจะนับ)
        results = model(image, conf=0.25)
        
        detections = []
        result = results[0]
        
        # 3. ดึงรายละเอียดของแต่ละกรอบ (Box) อย่างปลอดภัย
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            
            detections.append({
                "box": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(confidence, 2),
                "class_id": class_id
            })
        
        count = len(detections)
        
        # 4. แสดงผลในจอดำ (Terminal)
        print(f"📸 รับรูปภาพแล้ว -> 🐔 AI นับได้: {count} ตัว")
        
        return {
            "success": True,
            "count": count,
            "detections": detections, # 🔥 อัปเดตตรงนี้! ส่งพิกัดกลับไปให้ React วาดกรอบ
            "message": f"Found {count} chickens"
        }
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return {"success": False, "count": 0, "error": str(e)}

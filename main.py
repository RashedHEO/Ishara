import os
import fitz
import json
import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. تهيئة التطبيق ومفتاح الـ API
app = FastAPI(title="Real Estate Brochure API")
# سيتم قراءة المفتاح من بيئة السيرفر (آمن جداً)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. تحديد شكل البيانات المستلمة من زابيير
class BrochureRequest(BaseModel):
    pdf_url: str

# 3. دوال المعالجة (نفس كودك البطل مع تعديل بسيط للتعامل مع المسارات الديناميكية)
def process_pdf_multimodal(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = ""
    images_parts = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        links = page.get_links()
        urls = [link["uri"] for link in links if "uri" in link]
        
        extracted_text += f"\n--- صفحة {page_num+1} ---\n{text}"
        if urls:
            extracted_text += "\n[روابط مخفية]: " + ", ".join(urls) + "\n"
            
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = pix.tobytes("jpeg")
        images_parts.append(types.Part.from_bytes(data=img_data, mime_type="image/jpeg"))
        
    return extracted_text, images_parts

def analyze_with_vision(text_content, images_parts):
    system_prompt = """
    Task: Convert the provided brochure into a strictly formatted JSON array.
    
    Strict Filtering Logic (Exclude Sold/Reserved):
    - Before processing any unit, check for status keywords such as (محجوز, تم البيع, مباع, Sold, Reserved).
    - If a unit is marked with any of these statuses, DO NOT include it in the JSON array.
    
    Strict Mapping Logic for Riyadh Regions:
    Analyze the 'Neighborhood' (الحي) and output ONLY one of these four exact strings:
    شمال الرياض: (الملقا، الياسمين، النرجس، العارضة، حطين، الصحافة، الربيع، العقيق، الغدير).
    شرق الرياض: (اليرموك، المونسية، الرمال، القادسية، قرطبة، غرناطة، النهضة، الخليج).
    غرب الرياض: (لبن، طويق، نمار، الجبيلة، السويدي).
    جنوب الرياض: (العزيزية، الشفاء، بدر، الدار البيضاء).
    
    Output Fields & Data Integrity:
    1. Unit_Name: String format [Developer name] - [Project name] - [Unit Number].
    2. Property_Type: (شقة, دور, فلة, تاون هاوس, شقة دورين).
    3. Location: MUST match one of the 4 regions above.
    4. Price: Integer only. Remove "SAR", "ريال", commas, or spaces.
    5. Area: Float or Integer only. Remove "sqm", "متر", or any text.
    
    Return ONLY a valid [{},{}] array.
    """
    content_parts = [system_prompt, f"\nالنص المساعد:\n{text_content}"] + images_parts

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=content_parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    return json.loads(response.text)

# 4. نقطة الاتصال (The Endpoint)
@app.post("/extract-data")
async def extract_data(request: BrochureRequest):
    try:
        # تحميل الـ PDF من الرابط القادم من زابيير
        response = requests.get(request.pdf_url)
        response.raise_for_status()
        
        # حفظ الملف مؤقتاً في السيرفر
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(response.content)
            tmp_pdf_path = tmp_pdf.name

        # تشغيل محرك الذكاء الاصطناعي
        text_data, images = process_pdf_multimodal(tmp_pdf_path)
        final_result = analyze_with_vision(text_data, images)
        
        # تنظيف السيرفر من الملف المؤقت
        os.remove(tmp_pdf_path)
        
        return {"status": "success", "data": final_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
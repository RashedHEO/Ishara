import os
import fitz
import json
import tempfile
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Real Estate Brochure API")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=GEMINI_API_KEY)

class BrochureRequest(BaseModel):
    pdf_url: str

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
    # رجعنا تعليماتك الأصلية والصارمة 100% هنا
    system_prompt = """
    Task: Convert the provided brochure into a strictly formatted JSON array.

    Strict Filtering Logic (Exclude Sold/Reserved):
    - Before processing any unit, check for status keywords such as (محجوز, تم البيع, مباع, Sold, Reserved).
    - If a unit is marked with any of these statuses, DO NOT include it in the JSON array. Only extract available units.

    Strict Mapping Logic for Riyadh Regions:
    Analyze the 'Neighborhood' (الحي) and output ONLY one of these four exact strings for the Location field. If not found, output "".

    شمال الرياض: (الملقا، الياسمين، النرجس، العارضة، حطين، الصحافة، الربيع، العقيق، الغدير).
    شرق الرياض: (اليرموك، المونسية، الرمال، القادسية، قرطبة، غرناطة، النهضة، الخليج).
    غرب الرياض: (لبن، طويق، نمار، الجبيلة، السويدي).
    جنوب الرياض: (العزيزية، الشفاء، بدر، الدار البيضاء).

    Output Fields & Data Integrity:
    1. Unit_Name: String format [Developer name] - [Project name] - [Unit Number]. (Example: نصل العقارية - نصل 18 - 02).

    2. Property_Type: MUST match one of these exactly: (شقة, دور, فلة, تاون هاوس, شقة دورين). If ambiguous, output "".
    3. Location: MUST match one of the 4 regions above exactly based on the neighborhood.
    4. Price: Integer only. Remove "SAR", "ريال", commas, or spaces.
    5. Area: Float or Integer only. Remove "sqm", "متر", or any text.
    6. Project_Name: String format. The specific name of the project (Example: نصل 18).

    Constraint Checklist (Zero Tolerance):
    - No conversational filler. No "Here is the JSON".
    - Return ONLY a valid [{},{}] array.
    - If a value is missing, use "". NEVER hallucinate a price or location.
    - Ensure all Arabic strings are clean and free of hidden spaces.
    """
    
    content_parts = [system_prompt, f"\nالنص المساعد:\n{text_content}"] + images_parts

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=content_parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            return json.loads(response.text)
        except Exception as e:
            if "503" in str(e) and attempt < 2:
                print(f"زحمة في السيرفر، المحاولة رقم {attempt + 2} بعد 5 ثواني...")
                time.sleep(5)
                continue
            raise e

@app.post("/extract-data")
async def extract_data(request: BrochureRequest):
    try:
        response = requests.get(request.pdf_url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(response.content)
            tmp_pdf_path = tmp_pdf.name

        text_data, images = process_pdf_multimodal(tmp_pdf_path)
        final_result = analyze_with_vision(text_data, images)
        
        os.remove(tmp_pdf_path)
        
        return {"status": "success", "data": final_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
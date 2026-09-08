"""Interactive test script for Al-Mozare3 RAG Backend.

Runs realistic Egyptian farmer scenarios against the live /api/chat endpoint.
"""

import json
import sys
import requests

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

SCENARIOS = [
    {
        "title": "🌾 سيناريو 1: مزارع بنجر سكر في كفر الشيخ (مستند مفهرس بالفعل)",
        "payload": {
            "query": "ما هي أهم معاملات الري ومكافحة الحشائش في محصول بنجر السكر؟",
            "farmer_context": {
                "farmer_name": "الحاج أحمد العوضي",
                "crop": "بنجر السكر",
                "soil_type": "طينية",
                "irrigation_system": "ري بالغمر",
                "governorate": "كفر الشيخ",
                "crop_age_days": 35,
            },
        },
    },
    {
        "title": "🌾 سيناريو 2: مزارع قمح يسأل عن ري المحاياة والتسميد الآزوتي",
        "payload": {
            "query": "ما هي كمية السماد المطلوبة لري المحاياة في القمح وكيف يتم إضافته؟",
            "farmer_context": {
                "farmer_name": "المهندس محمود",
                "crop": "قمح",
                "soil_type": "طينية خصبة",
                "irrigation_system": "ري بالغمر",
                "governorate": "الدقهلية",
                "crop_age_days": 25,
            },
        },
    },
    {
        "title": "🛑 سيناريو 3: اختبار منع الهلوسة (سؤال خارج النشرات المعتمدة)",
        "payload": {
            "query": "كيف أزرع أشجار البن والشاي في الفضاء الخارجي بدون جاذبية؟",
            "farmer_context": {
                "farmer_name": "سعيد",
                "crop": "بن",
                "governorate": "الوادي الجديد",
            },
        },
    },
]


def test_scenario(index: int, item: dict):
    print("=" * 80)
    print(f"▶️  {item['title']}")
    print("=" * 80)

    payload = item["payload"]
    print(f"❓ السؤال: {payload['query']}")
    if "farmer_context" in payload:
        fc = payload["farmer_context"]
        print(f"👨‍🌾 المزارع: {fc.get('farmer_name')} | المحصول: {fc.get('crop')} | المحافظة: {fc.get('governorate')}")
    print("\n⏳ جاري إرسال الطلب إلى الخادم واسترجاع الإجابة عبر Gemini...")

    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            print("\n✅ الإجابة المستلمة من المستشار الزراعي:")
            print("-" * 50)
            print(data.get("answer", ""))
            print("-" * 50)

            sources = data.get("sources", [])
            print(f"📚 المصادر المستند إليها ({len(sources)} مصادر):")
            for idx, s in enumerate(sources, 1):
                page_str = f" (صفحة {s['page']})" if s.get("page") else ""
                print(f"  [{idx}] 📄 {s['document']}{page_str}")
                print(f"       اقتباس: {s['snippet'][:120]}...")
        else:
            print(f"❌ خطأ من الخادم (كود {response.status_code}):")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ تعذر الاتصال بالخادم! تأكد من أن الخادم يعمل بالأمر:")
        print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ خطأ أثناء تنفيذ الاختبار: {e}")
    print("\n")


if __name__ == "__main__":
    print("🌾 بدء تشغيل سيناريوهات اختبار باك إند المزارع (Al-Mozare3 RAG Tester) 🌾\n")
    for i, scenario in enumerate(SCENARIOS, 1):
        test_scenario(i, scenario)

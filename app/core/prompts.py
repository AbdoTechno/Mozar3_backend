from typing import List, Optional
from app.schemas.rag_schemas import ChatMessage, FarmerContext

SYSTEM_PROMPT = """أنت "المستشار الزراعي الذكي لتطبيق المزارع (Al-Mozare3)"، مهندس وخبير زراعي مصري متميز متخصص في الإرشاد الزراعي المصري وإدارة المحاصيل وفق النشرات الرسمية لوزارة الزراعة ومركز البحوث الزراعية.

### مهامك وقواعد إجابتك:
1. **تكييف الرد والأسلوب حسب صفة السائل (User Role Persona)**:
   - **إذا كان السائل (مزارع / فلاح)**:
     تحدث بلغة مصرية فصحى ميسرة جداً بالاصطلاحات الدارجة والمحببة للفلاح (ري المحاياة، الشيكارة، التكبيش، التسميد، العزيق، الشتل، فدان، قيراط). ركز على الخطوات الميدانية المباشرة والجرعات بالأكياس والكيلوجرامات وابتعد عن المصطلحات الكيميائية المعقدة.
   - **إذا كان السائل (مهندس زراعي / استشاري)**:
     قدم إجابة علمية متقدمة ودقيقة، اذكر المواد الفعالة (Active Ingredients)، والنسب المئوية، وميكانيزم التأثير، وأسماء الآفات العلمية وفق بروتوكولات مركز البحوث الزراعية.
   - **إذا كان السائل (تاجر / مستثمر زراعي)**:
     ركز على مواصفات الجودة، درجات النضج، معاملات ما بعد الحصاد والتبريد، التخزين الأمثل، وتفادي الفاقد التسويقي.
   - **إذا كان السائل (هاوٍ / مبتدئ)**:
     اشرح المفاهيم بالتدريج وبأسلوب تعليمي مبسط خطوة بخطوة مع توضيح أي اصطلاح زراعي.

2. **الاعتماد الصارم على النشرات المرفقة (Context)**:
   - اجعل إجابتك مستندة حصراً ودقة إلى المعلومات المرفقة في قسم (المعلومات والنشرات المسترجعة).
   - اذكر الجرعات، التوقيتات، ونسب التسميد أو الرش بدقة وأمان تام.
   - لا تؤلف أو تبتكر أرقاماً أو أسماء مبيدات من تلقاء نفسك خارج النشرات المرفقة.

3. **تخصيص الإجابة حسب سياق المزارع (Farmer Context)**:
   - إذا تم تزويدك بمعلومات عن المزارع (مثل المحصول، نوع التربة، نظام الري، المحافظة، وعمر المحصول بالأيام)، قم بتفصيل التوصية لتناسب هذه الظروف تحديداً (مثلاً: التربة الرملية تحتاج تنظيم فترات الري والتسميد بالتنقيط مقارنة بالأرض الطينية، ومرحلة عمر 40 يوم تختلف عن بداية الزراعة).

4. **قاعدة عدم وجود المعلومة (Strict Grounding Fallback)**:
   - إذا كان السؤال يخص موضوعاً لم يرد بشكل كافٍ أو دقيق في النشرات المرفقة، التزم بالعبارة التالية نصاً في بداية ردك:
     "بناءً على النشرات المعتمدة المتوفرة حالياً، لم أجد إجابة دقيقة على هذا السؤال، ولكن أنصحك بـ..."
   - ثم قدم نصيحة إرشادية عامة آمنة واقترح عليه التوجه للمرشد الزراعي بالجمعية الزراعية في قريته لمراجعة الحالة عملياً.

5. **تنسيق الرد**:
   - استخدم نقاطاً مرتبة، خطوات عملية، وإرشادات واضحة ليسهل على السائل قراءتها أو الاستماع إليها.
"""


def build_farmer_context_text(farmer_context: Optional[FarmerContext]) -> str:
    """Format farmer context into a readable summary string."""
    if not farmer_context:
        return "لا توجد بيانات محددة للمزارع (إجابة عامة لجميع المزارعين)."

    role = farmer_context.user_role or "مزارع"
    details = [f"- صفة/رتبة السائل (Role): {role}"]

    if farmer_context.farmer_name:
        details.append(f"- الاسم: {farmer_context.farmer_name}")
    if farmer_context.crop:
        details.append(f"- المحصول المنزرع: {farmer_context.crop}")
    if farmer_context.soil_type:
        details.append(f"- نوع التربة: {farmer_context.soil_type}")
    if farmer_context.irrigation_system:
        details.append(f"- نظام الري: {farmer_context.irrigation_system}")
    if farmer_context.governorate:
        details.append(f"- المحافظة / المنطقة: {farmer_context.governorate}")
    if farmer_context.crop_age_days is not None:
        details.append(f"- عمر المحصول الحالي: {farmer_context.crop_age_days} يوم")

    return "\n".join(details)


def build_rag_prompt(
    query: str,
    context_chunks: List[str],
    farmer_context: Optional[FarmerContext] = None,
    history: Optional[List[ChatMessage]] = None,
) -> str:
    """Assemble final prompt for Gemini LLM combining context, profile, history, and query."""

    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "لا توجد نصوص وثائق مسترجعة ذات صلة قريبة."
    farmer_info = build_farmer_context_text(farmer_context)
    role = farmer_context.user_role if farmer_context and farmer_context.user_role else "مزارع"

    history_text = ""
    if history:
        history_lines = []
        for msg in history[-6:]:  # Keep last 3 exchanges for conciseness
            role_label = "المستخدم" if msg.role == "user" else "المستشار الزراعي"
            history_lines.append(f"{role_label}: {msg.content}")
        history_text = "\n### سجل المحادثة السابقة:\n" + "\n".join(history_lines) + "\n"

    prompt = f"""### معلومات وظروف السائل وحقله:
{farmer_info}
{history_text}
### المعلومات والنشرات المسترجعة من النشرات الرسمية (Context):
{context_text}

### سؤال السائل الحالي:
{query}

### توجيه للإجابة:
أجب عن السؤال متوافقاً تماماً مع صفة السائل ({role})، فإن كان مزارعاً فخاطبه ببساطة وبمصطلحات الحقل العملية وجرعات الأكياس، وإن كان مهندساً فأعطه العمق الفني والعلمي. اعتمد بدقة على النشرات المرفقة أعلاه، مراعياً تعليمات السلامة والجرعات بدقة.
"""
    return prompt

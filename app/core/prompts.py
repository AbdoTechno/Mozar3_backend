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

2. **الاعتماد الأساسي على النشرات المرفقة (Context)**:
   - اجعل إجابتك مستندة بالدرجة الأولى إلى المعلومات المرفقة في قسم (المعلومات والنشرات المسترجعة).
   - اذكر الجرعات، التوقيتات، ونسب التسميد أو الرش بدقة وأمان تام وفق الإرشادات الرسمية المعتمدة.

3. **تخصيص الإجابة حسب ظروف المزارع وحقله (Farmer Context)**:
   - استفد دائماً من بيانات السائل وحقله (المحصول، نوع التربة، نظام الري، المحافظة، وعمر النبات بالأيام)، وفصّل التوصية لتناسب أرضه تماماً (مثلاً: طبيعة التربة الرملية تختلف عن الطينية في تقسيط الري والتسميد).

4. **الاستنتاج الزراعي الذكي وتقديم أقرب نصيحة (Smart Grounding & Nearest Advice)**:
   - إذا لم تجد تفصيلة معينة بالنص الحرفي في النشرات المرفقة، **لا تكتفِ بالاعتذار أو الرفض الجاف**؛ بل استخدم خبرتك الإرشادية لتقديم **أقرب نصيحة زراعية عملية ومفيدة** تناسب حالة المحصول ونظام الري ونوع التربة في مصر، مع تنبيه المزارع في نهاية الرد لمراجعة المرشد الزراعي في الجمعية الزراعية لتأكيد الجرعة عملياً.

5. **تنسيق الرد**:
   - استخدم نقاطاً مرتبة، خطوات عملية واضحة، وإرشادات حقلية مباشرة تريح المزارع وتجيب على سؤاله بدقة.

6. **التعامل مع فحص الكاميرا وعينات الأمراض (Camera & Disease Diagnosis)**:
   - إذا سأل المزارع عن مرض نباتي أو ذكر أنه فحص عينة ورقية بالكاميرا (مثل: فراولة، طماطم، بطاطس، عنب، فلفل، خوخ... إلخ):
     * **يُمنع منعاً باتاً الاعتراض على نوع المحصول أو الادعاء بأن الكاميرا أخطأت لمجرد أن محصول حيازة المزارع المسجل مختلف (مثل الأرز أو القمح)**؛ فالمزارع يفحص عينات لمساحات خضرية ثانوية أو حدائق أو أراضي جيرانه.
     * **أجب فوراً وبالكامل عن المحصول والمرض المفحوص المذكور في السؤال**، وقدم خطة علاجية مصرية متكاملة تشمل:
       أ. تشخيص المرض ومسببه الفطري/البكتيري.
       ب. أسماء المبيدات التجارية الموصى بها في مصر والمادة الفعالة (Active Ingredient).
       ج. جرعات الرش الدقيقة لكل 100 لتر ماء وفترة الأمان قبل الحصاد (PHI).
       د. الإجراءات الوقائية وإرشادات الري والتسميد لتعافي النبات.
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

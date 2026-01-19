# 🚀 AI Inference Parallelization Project

## 📋 نظرة عامة / Overview

مشروع لقياس وتحسين أداء عمليات الاستدلال (Inference) في نماذج الذكاء الاصطناعي باستخدام تقنيات التوازي (Parallelization). يقارن المشروع بين التنفيذ المتسلسل (Sequential) والتنفيذ متعدد الخيوط (Multi-threaded) باستخدام نموذج ResNet18.

A project to measure and optimize AI model inference performance using parallelization techniques. The project compares sequential execution with multi-threaded execution using ResNet18 model.

---

## 🎯 الهدف من المشروع / Project Objectives

- **مقارنة الأداء**: مقارنة بين التنفيذ المتسلسل والتنفيذ المتوازي
- **تحسين الأداء**: استخدام Multi-threading لتسريع عمليات الاستدلال
- **القياس والتحليل**: توليد رسوم بيانية لمقارنة الأداء

- **Performance Comparison**: Compare sequential vs parallel execution
- **Performance Optimization**: Use multi-threading to accelerate inference operations
- **Measurement & Analysis**: Generate performance comparison graphs

---

## 📁 هيكل المشروع / Project Structure

```text
AI-Inference-Parallelization/
├── .github/
│   └── workflows/
│       └── benchmark.yml          # GitHub Actions workflow
├── main.py                        # الكود الرئيسي / Main script
├── AI-Inference-Parallelization.pdf  # التقرير / Report
└── README.md                      # هذا الملف / This file
```

---

## 🛠️ المتطلبات / Requirements

### المكتبات المطلوبة / Required Libraries

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### تثبيت المتطلبات / Installation

#### للاستخدام المحلي (مع GPU/CPU قوي) / For Local Use:

```bash
pip install torch torchvision numpy matplotlib
```

#### للاستخدام على خوادم محدودة (مثل GitHub Actions) / For CPU-only servers:

```bash
pip install numpy matplotlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🚀 الاستخدام / Usage

### تشغيل البرنامج محلياً / Run Locally

```bash
python main.py
```

### الإعدادات القابلة للتعديل / Configurable Settings

في ملف `main.py`، يمكنك تعديل المعاملات التالية:

In `main.py`, you can modify the following parameters:

```python
NOMBRE_INFERENCES = 100      # عدد عمليات الاستدلال / Number of inferences
NOMBRE_THREADS = 8           # عدد الخيوط المستخدمة / Number of threads
DIM_INPUT = (1, 3, 224, 224) # أبعاد البيانات المدخلة / Input dimensions
```

---

## 🔄 GitHub Actions Workflow & Results

المشروع يحتوي على workflow تلقائي يعمل على GitHub Actions عند كل Push أو Pull Request.

The project includes an automated workflow that runs on GitHub Actions on every Push or Pull Request.

### ⚠️ تحليل الأداء في بيئة CI/CD / Performance Analysis on CI/CD

🛑 **ملاحظة مهمة حول النتائج في GitHub Actions**: قد تلاحظ أن الفرق في السرعة بين التنفيذ المتسلسل والمتوازي ضئيل جداً (أو معدوم) في تقرير GitHub Actions.

**السبب التقني**: خوادم GitHub Actions المجانية تعمل بـ 2 vCPUs فقط. عندما نحاول تشغيل 8 Threads، يضطر المعالج لقضاء وقت طويل في التبديل بين المهام (Context Switching)، مما يستهلك الموارد ويلغي فائدة التوازي.

**الخلاصة**: التوازي يظهر كفاءته الحقيقية على الأجهزة المحلية (Local Machines) التي تحتوي على عدد أنوية أكبر (4+ Cores).

🛑 **Important Note on GitHub Actions Results**: You might notice minimal speedup differences in the GitHub Actions report.

**Technical Explanation**: Free GitHub Actions runners are strictly limited to 2 vCPUs. Launching 8 Threads on a dual-core system forces excessive Context Switching, creating overhead that negates parallelization benefits.

**Conclusion**: Parallelization efficiency is best demonstrated on local machines with higher core counts (4+ Cores).

---

## 📊 المخرجات / Outputs

بعد تشغيل البرنامج، ستحصل على:

After running the script, you will get:

1. **رسالة في وحدة التحكم** / **Console Output**:
   - وقت التنفيذ المتسلسل / Sequential execution time
   - وقت التنفيذ المتوازي / Parallel execution time
   - نسبة التسريع (Speedup) / Speedup ratio

2. **رسم بياني** / **Performance Graph**:
   - ملف `performance_graph.png` يتم توليده تلقائياً.
   - A `performance_graph.png` file is automatically generated.

---

## 📚 الوثائق / Documentation

للحصول على تفاصيل أكثر حول المشروع، والنتائج النظرية، راجع ملف التقرير المرفق:

For more details about the project and theoretical results, see the attached PDF:

📄 [AI-Inference-Parallelization.pdf](AI-Inference-Parallelization.pdf)

---

## 🔧 التقنيات المستخدمة / Technologies Used

- **PyTorch**: للتعلم العميق وإدارة النماذج / Deep Learning & Model Management
- **ResNet18**: نموذج التصنيف المدرب مسبقاً / Pre-trained Classification Model
- **ThreadPoolExecutor**: لإدارة التوازي / For Parallel Execution
- **Matplotlib**: لتصوير البيانات / For Data Visualization

---

## 👥 فريق العمل / Authors

- **Amanetoullah** (C22643)
- **Hashimi** (C21454)

---

## 📝 الترخيص / License

هذا المشروع متاح للاستخدام التعليمي والبحثي.

This project is available for educational and research purposes.

---

**⭐ إذا أعجبك المشروع، لا تنسى إضافة نجمة! / If you like this project, don't forget to add a star! ⭐**

# Sensor-Based Human Activity Recognition & Educational Intervention Analytics (`har-and-educational-analytics`)

An integrated computational framework uniting **wearable sensor kinematic feature extraction for Human Activity Recognition (HAR)** and **multivariate survey-based modeling of educational analytics and student interventions**, drawing directly from graduate machine learning research at **PAF-IAST** and academic leadership at the **University of Haripur**.

## Author & Academic Profile
- **Author & Maintainer:** **Rabia Khan**
- **Current Appointment:** Lecturer, Department of Information Technology, University of Haripur, Khyber Pakhtunkhwa, Pakistan
- **Academic Credentials:**
  - **M.Sc. in Data Science**, Pak-Austria Fachhochschule: Institute of Applied Sciences & Technology (PAF-IAST), Haripur
  - **B.Sc. in Computer Engineering**, COMSATS University Islamabad, Abbottabad Campus
- **ORCID:** `0009-0007-5575-1564`

## Modules & Empirical Results
1. **Wearable Sensor Feature Extraction (`SensorFeatureExtractor`):** Tri-axial accelerometer and gyroscope time- and frequency-domain statistics, Signal Magnitude Area (SMA), and spectral entropy (`26 features / window`).
2. **HAR Physical Activity Classifier (`HARClassifier`):** Evaluates 5 physical activity states (`WALKING`, `UPSTAIRS`, `DOWNSTAIRS`, `SITTING`, `STANDING`), achieving `95.0%` accuracy and `0.9500` Macro-F1.
3. **Educational Retention & Intervention Analytics (`EducationalRetentionModel`):** Multivariate survey and LMS engagement modeling achieving `1.0000` AUC and identifying key intervention odds ratios.

## Quickstart
```bash
pip install -e .
python -m unittest discover -s tests -v
python -m har_edu_analytics.cli
```

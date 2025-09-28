#  Gas Fault Prediction System  

##  Overview  
This project was developed for the **Digitopia University Students Competition – Innovator Impact Track**.  
It delivers an **AI-powered early fault detection system** for gas pipelines, capable of predicting **system warnings and failures 2 hours in advance**.  

The solution includes:  
- 📊 **AI Model** trained on SCADA-like industrial data.  
- 🖥️ **Interactive Dashboard** for visual monitoring of system status.  
- 🤖 **Smart Chatbot** to support engineers with quick, natural-language insights.  

---

##  Problem Definition  
Gas distribution systems are **critical national infrastructure**. Failures can lead to:  
- Safety hazards for workers and civilians  
- Expensive downtime and energy loss  
- Disruption of industrial operations  

**Our goal**: build a predictive system that **anticipates failures and warnings before they happen**, enabling preventive actions and saving both costs and lives.  

---

##  Methodology  

1. **Synthetic Data Generation**  
   - Designed datasets that mimic **real SCADA signals** (pressure, flow, temperature, pump speed, energy consumption).  
   - Extended to cover **7 days of continuous operation**, ensuring realistic event sequences.  

2. **Feature Engineering**  
   - Lag features to capture historical patterns.  
   - Rolling window statistics (mean, std, min, max).  
   - Time features (`hour`, `dayofweek`) for temporal patterns.  

3. **Model Training**  
   - Algorithms: **LightGBM, CatBoost, XGBoost**.  
   - Tackled class imbalance using **SMOTE** and **class weights**.  
   - Ensemble voting classifier + parameter tuning for robustness.  

4. **Results**  
   - **Accuracy:** 77.3%  
   - **Class-level performance:**  
     - Normal → F1 = 0.80  
     - Fault → F1 = 0.71  
     - Critical → F1 = 0.85  
   - Key achievement: **high recall in predicting warnings and failures**, which was the system’s main goal.  

5. **System Deployment**  
   - **Dashboard:** real-time scenario simulation (Normal, Warning, Failure).  
   - **Chatbot:** connected to dashboard predictions via Google Sheets API, giving engineers a **faster alternative** to manual dashboard monitoring.  

---

##  Team Roles  
- **Tasneem:** Data generation, feature engineering, AI model training, dashboard architecture.  
- **Ahmad:** Exploratory data analysis (EDA), chatbot architecture, testing, final integration.  
- **Kareem:** UML diagrams, ERD design, database modeling.  
- **Yousef:** Video preparation, linking dashboard results with chatbot via API.  

---


---

##  Report  
Full details are provided in the report 👉 [Report.pdf](report/digi.pdf)  

---

##  Future Vision  
- Direct integration with **real SCADA pipelines** (replace scenario simulation).  
- Extend prediction horizon beyond **2 hours**.  
- Deploy on **cloud infrastructure** for scalability.  
- Integrate with **mobile apps** for instant engineer alerts.  

---

##  Impact  
This project shows how **AI can safeguard Egypt’s energy infrastructure** by predicting faults before they occur.  
It represents a scalable business model that can be **seamlessly integrated into existing SCADA systems** with minimal changes.  


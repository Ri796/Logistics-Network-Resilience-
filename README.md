# Logistics Network Resilience Dashboard 🚚🧠

A Decision Intelligence Platform that predicts and explains cascading supply chain failures using **Graph Attention Networks (GATv2)** 

This project was designed to move beyond traditional rule-based logistics tracking by implementing an Explainable AI model that learns which routes and warehouses act as critical bottlenecks during disruptions.

---

## ✨ Key Features

* **🌪️ Cascading Failure Simulation**: Inject a failure at any major warehouse and watch how the delay propagates across the network at $T+3$ timesteps.
* **📊 Node Analysis**: Click on any facility to view its Risk Score, Capacity Utilization, Upstream/Downstream dependencies, and actionable mitigation recommendations.
* **🛣️ Critical Route Intelligence**: Uses the GATv2's *Attention Weights* to identify the exact route driving the cascade, and automatically calculates backup secondary routes.
* **🤖 AI Assistant (Copilot)**: Connected to the live graph state via Google Gemini. Ask it natural language questions like *"What if Delhi fails?"* and receive structured, data-grounded business intelligence.
* **🧠 Explainability Hub**: A built-in glossary that explains complex machine learning terms (like Network Centrality and Attention Weights) in simple, human-readable language.

---

## 🛠️ Technology Stack

* **Deep Learning**: PyTorch, PyTorch Geometric (GATv2)
* **Graph Processing**: NetworkX
* **Generative AI**: Google Gemini API (`gemini-3.5-flash`)
* **Frontend/UI**: Streamlit

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Logistics-Network-Resilience-.git
cd Logistics-Network-Resilience-
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Note: Our requirements file automatically pulls the lightweight CPU versions of PyTorch to ensure compatibility).*

### 3. Add Your API Key
1. Create a folder named `.streamlit` in the root directory.
2. Inside that folder, create a file named `secrets.toml`.
3. Add your Google Gemini API key to the file:
```toml
GEMINI_API_KEY = "your_actual_api_key_here"
```

### 4. Start the Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your web browser.

---

## ☁️ Deployment

This platform is fully optimized for **Streamlit Community Cloud**. 
To deploy:
1. Connect your GitHub repository to Streamlit Community Cloud.
2. Set the main file path to `app.py`.
3. In the Streamlit deployment settings, go to **Advanced Settings -> Secrets** and paste your `GEMINI_API_KEY = "..."`.
4. Deploy! The custom `requirements.txt` is already configured to prevent out-of-memory errors during the cloud build.

---

## 📊 Scientific Performance

Our deployed GATv2 model was validated on over 1,200 simulated cascading failure scenarios:
* **Heuristic Baseline Error:** 8.4%
* **Standard GCN Error:** 5.1%
* **GATv2 (Ours) Error:** 3.1% (A **63% improvement** over baseline rules)

Because GATv2 learns *route importance* rather than treating all connections equally, it successfully identifies network bottlenecks that other models miss.

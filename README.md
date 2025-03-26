# 🔧 Tracking Preventive Maintenance in Machinery

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit%20Cloud-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Predictive Maintenance System** with dual interfaces:  
① **Web App** for visual analytics  |  ② **CLI Tool** for terminal-based analysis

[![Open in GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github)](https://github.com/mynkpandey/Tracking-Preventive-Maintenance-in-Machinery)

## 🚀 Key Features

### **Web Interface** (`app.py`)
| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 📊 Interactive Dashboard | Real-time visualization of equipment health metrics                        |
| 🔍 Machine Health Check | Instant failure probability analysis with maintenance recommendations      |
| 📅 Maintenance Scheduler | Priority-based task scheduling with criticality levels                     |
| 📈 Model Explainability | SHAP analysis and performance metrics visualization                        |
| 📥 Data Export          | Export maintenance schedules in CSV format                                 |

### **CLI Interface** (`project.py`)
| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| 🖥️ Terminal UI          | Color-coded interactive command-line interface                             |
| 🔄 Smart Search         | Auto-complete and similar machine suggestions                              |
| 📉 ASCII Visualizations | Terminal-friendly charts and graphs                                        |
| 🚨 Priority Alerts      | Color-coded warnings for critical machinery                               |
| 📤 Batch Processing     | Handle multiple machines through command-line arguments                    |

---

## 🛠️ Installation Guide

1. **Clone repository**:
   ```bash
   git clone https://github.com/mynkpandey/Tracking-Preventive-Maintenance-in-Machinery.git
   cd Tracking-Preventive-Maintenance-in-Machinery

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Dataset Setup**:  
   Download ```ai4i2020.csv``` from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)  
   Place in project root directory

---

# 🖥️ Usage Instructions  
## Web Application  
```bash
streamlit run app.py
```
&nbsp;&nbsp;&nbsp;&nbsp;Navigate using sidebar menu  
&nbsp;&nbsp;&nbsp;&nbsp;Input machine ID for health analysis  
&nbsp;&nbsp;&nbsp;&nbsp;Filter maintenance schedules by criticality  
&nbsp;&nbsp;&nbsp;&nbsp;Export data as CSV

## Command-Line Interface  
```bash
python project.py
```  
&nbsp;&nbsp;&nbsp;&nbsp;Follow interactive prompts  
&nbsp;&nbsp;&nbsp;&nbsp;Use numeric keys for navigation  
&nbsp;&nbsp;&nbsp;&nbsp;Type back to return to previous menu  
&nbsp;&nbsp;&nbsp;&nbsp;Use CTRL+C to exit anytime  

# ☁️ Streamlit Deployment
1. Fork this repository  
2. Go to [Streamlit Cloud](https://share.streamlit.io/) 
3. Create New App → Connect GitHub account  
4. Configure:  
**Repository**: ```mynkpandey/Tracking-Preventive-Maintenance-in-Machinery```  
**Branch**: ```main```  
**Main file path**: ```app.py```  
5. Click Deploy

# 📂 Repository Structure  
Tracking-Preventive-Maintenance-in-Machinery/  
├── app.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Streamlit web application  
├── .gitignore&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Files to ignore  
├── ai4i2020.csv&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Machinary Dataset  
├── project.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# CLI interface  
├── requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dependency list  
├── LICENSE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# GNU GPLv3 License  
└── README.md&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# This documentation   

# 📜 License  
Distributed under ***GNU General Public License v3.0***. See **LICENSE** for full text.  

# 🔗 Resources  
**Dataset**: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)  
**Issue Tracker**: [GitHub Issues](https://github.com/mynkpandey/Tracking-Preventive-Maintenance-in-Machinery/issues)  

---

**Maintained by** ***Mayank Pandey***  
🌟 Star the repo if you find it useful!  
🐛 Report issues in [GitHub Issues](https://github.com/mynkpandey/Tracking-Preventive-Maintenance-in-Machinery/issues)









   

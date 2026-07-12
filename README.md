# HR Digital Twin: Gamifying Workforce Strategy

https://hr-digital-twin-de6ppkvxau3uvyzwzsb6oc.streamlit.app/

Welcome to the HR Digital Twin project! This repository contains the code and methodology for my Master's thesis, which reimagines Human Resources analytics. 

Most HR departments operate in the rearview mirror. They rely on static regression models and historical dashboards to explain why people quit *after* the fact. This project asks a different question: what if we could build a risk-free sandbox to test HR interventions before spending real money? 

This project is a Digital Twin of the Organization (DTO). It is an agent-based simulation that treats current employees as dynamic characters. By feeding real-time (and simulated) data into a machine learning engine, this tool allows management to visualize the workforce, predict attrition, gauge promotion readiness, and actively test how budget changes impact the future.

### The Engine Under the Hood
At its core, this isn't just a prediction script; it is a time-step simulation. 

Because we cannot hook directly into a live corporate CRM for a thesis project, I built a Synthetic Data Streaming Architecture. The Python backend acts as the "game clock." It feeds employee records into our machine learning models week-by-week, artificially aging variables like tenure and burnout accumulation.

To handle the predictions, the engine uses Random Forest and XGBoost algorithms. Crucially, because employee attrition datasets suffer from massive class imbalance (thankfully, most people don't quit every day), the data pipeline incorporates SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model doesn't just lazily predict that everyone stays.

### The Hybrid Data Strategy
To make the simulation both behaviorally accurate and financially realistic, the engine fuses two different types of open-source data:

**The Logic Layer:** We use the IBM HR Analytics and Dr. Rich Huebner datasets to teach the AI how people behave. This gives the model vital social dynamics, allowing it to understand how factors like distance from home or a manager's performance score impact an employee's flight risk.

**The Realism Layer:** To make the game economy reflect the real world, the simulation is injected with data from the San Francisco City Employee Salary Database and US OPM FedScope. This populates the simulation with authentic job titles, overtime pay structures, and salary brackets.

### Key Simulation Metrics (KPIs)
The simulation tracks two distinct sets of metrics to balance the needs of the algorithm and the HR manager playing the simulation.

**Agent Health Bars (Model Inputs):**
Every simulated employee has underlying stats driving their behavior. The model calculates a real-time Flight Risk Score and a Promotion Readiness Score. It also tracks a Stagnation Index (how long they've been stuck in one role), a Burnout Meter driven by overtime, and Manager Friction to simulate how bad managers impact team morale.

**The Management Scoreboard (Model Outputs):**
The end-user interacts with the macro results. As the simulation runs, managers monitor the Regrettable Attrition Cost—a real-time financial ticker showing the exact dollar amount lost when high-performers leave. They can also track the overall health of the Talent Pipeline and the simulated mood of the office (eNPS). 

### Current Status and Next Steps
The core machine learning pipeline and time-step generator are currently implemented. The immediate next phase is expanding the front-end visualization so non-technical users can interact with the budget sliders and watch the simulation unfold.

Feel free to explore the code, fork the repository, or reach out if you want to chat about the intersection of Data Analytics and HR Strategy!

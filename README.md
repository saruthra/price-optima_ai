# Price Optima AI 

A full-stack dynamic pricing application for a ride-sharing service. This project uses a machine learning model served by a FastAPI backend and a responsive React frontend to provide real-time price recommendations.

This repository contains the complete code for both the frontend and backend services.

## Features

* **Dynamic Price Prediction:** Uses a scikit-learn model to predict optimal ride prices based on various factors.
* **Interactive Frontend:** A clean React UI to input ride details and receive price recommendations.
* **RESTful API:** A high-performance backend built with FastAPI to serve the model and business logic.
* **Full-Stack Architecture:** Clear separation between the `backend` (API) and `frontend` (client) applications.

##  Tech Stack

* **Backend:**
    * **Python 3.13+**
    * **FastAPI:** For the API server.
    * **Scikit-learn:** For using the machine learning model.
    * **Pandas & NumPy:** For data manipulation.
    * **Uvicorn:** As the ASGI server.
* **Frontend:**
    * **React.js**
    * **JavaScript (ES6+)**
    * **Axios:** For making API calls to the backend.
    * **CSS:** For custom styling.

##  Getting Started

To run this project locally, you will need to run both the backend and frontend servers in separate terminals.

### Prerequisites

* **Node.js** (v18 or newer) & **npm**
* **Python** (v3.13 or newer) & **pip**
* **Git**

---

### 1. Backend Setup

First, set up and run the FastAPI server.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/saruthra/price-optima_ai.git](https://github.com/saruthra/price-optima_ai.git)
    cd price-optima_ai
    ```

2.  **Navigate to the backend folder:**
    ```bash
    cd backend
    ```

3.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    # On macOS/Linux: source venv/bin/activate
    ```

4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the backend server:**
    ```bash
    uvicorn main:app --reload
    ```
    🎉 Your backend is now running at `http://localhost:8000`

---

### 2. Frontend Setup

In a **new, separate terminal**, set up and run the React app.

1.  **Navigate to the frontend folder** (from the root `price-optima_ai` directory):
    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Run the frontend server:**
    ```bash
    npm start
    ```
    Your frontend is now running and will automatically open at `http://localhost:3000` in your browser.

---

## How to Use

Once both servers are running:
1.  Open `http://localhost:3000` in your browser.
2.  Fill in the "Ride Details" form with your desired inputs.
3.  Click the **"Get Optimal Price"** button.
4.  The frontend will send a request to the backend, which will return the predicted price to the "Recommended Price" card.

##  Jupyter Notebook

This repository also includes the original Colab notebook, `PRICE_OPTIMA.ipynb`, which details the data exploration, feature engineering, and model training process.

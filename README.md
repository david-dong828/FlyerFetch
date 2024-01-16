# FlyerFetch Application

**Introduction** 

FlyerFetch is an innovative application designed to save time and enhance the shopping experience for users who frequently consult store flyers. Traditional flyers are often cumbersome and time-consuming to go through. FlyerFetch addresses this issue by digitizing flyer data and offering personalized recommendations.

------
**Purpose**

- **Time Efficiency**: Reduces the time spent reading through traditional paper flyers.
- **Convenience**: Presents flyer information in a concise, digital format.
- **Personalized Recommendations**: Utilizes NLP and machine learning to offer tailored shopping suggestions.

------

> **Technology Stack**

- **Framework**: Flask (Python-based web framework)
- **Data Scraping**: Selenium for automated flyer data extraction from various sources.
- **Data Processing**: Python scripts for cleaning and organizing flyer data.
- **Recommendation Engine**: Utilizes NLP and combined algorithms for generating personalized recommendations.
- 
----
> **Process**
- **Data Scraping**: Automated scripts use Selenium to scrape flyer data from designated sources.
- **Data Cleaning**: Extract and clean data to remove duplicates and irrelevant information. Key details such as item name, price, and unit of measurement are retained.
- **Recommendation Calculation**: Apply NLP techniques and custom algorithms to analyze the data and compute personalized recommendations.
- **Server**: Flask server hosts the application, providing a user interface for interaction.

-----

> **Running the Application**
- **Initial Setup**: Ensure Python and required packages (Flask, Selenium, etc.) are installed.
- **Start the Application**: Run main.py to start the Flask server.
- **Access the Web Interface**: Open a web browser and navigate to the local server address (typically localhost with a designated port).
  - **First-Time Use**: The initial run may take longer due to the absence of a local data cache. The application will scrape and process data during this time.
  - **Subsequent Use**: Data processing is faster after the initial setup, as the application utilizes locally cached data.
 
- **Note:**
  - The model "model_epoch_13_11.pt" used for recommendation engine is too big to upload Github
  - You may want to train the model by yourself. If so, you could run the single file *"BiLSTMWithXLMRModel.py"* to get your trained model.

------

> **Result**


<!--- https://github.com/david-dong828/FlyerFetch/assets/106771290/5b659e0b-da0b-433c-a3d0-f97346725656 -->

https://github.com/david-dong828/FlyerFetch/assets/106771290/2f093410-fa52-4061-876f-ffe16ad4517c



------

> **NEXT UPDATE**

- [x] Upgrade the Classification model
      - LTSM + XLMRobertaModel

- [ ] Add more groceries



# Car Workshop "Pimp My Wheels" Database Project

## Project Overview

Car Workshop Database Project is a comprehensive database management system for a car workshop, aimed at handling various aspects such as staff, equipment, vehicles, transactions, and customers. This project covers the entire lifecycle of database creation and management, from design and data generation to analysis and reporting.

The project is divided into five key parts:

1. **Database Design** – Creating a schema to manage workshop resources, including staff, equipment, vehicles, transactions, and customers.
2. **Data Generation** – Scripting realistic and representative data for the database, using tools like Python to automate the process.
3. **Data Analysis** – Analyzing trends related to repairs and sales to derive meaningful insights for profitability and operational efficiency.
4. **Report Generation** – Automatically producing reports from the analysis, including visualizations and insights, in a PDF format.
5. **Documentation** – Providing details about the project, the technologies used, and instructions for running the project.

While my main focus was on the **Database Design** and **Data Generation**, I also contributed to all other aspects of the project.

## Project Structure

- `data/`                   : Contains the raw and processed data for the project
- `models/`                 : Holds any models related to the project (e.g., data models)
- `fill_database.py`        : Script to populate the database with realistic data
- `config.py`               : Configuration file for database connection and settings
- `requirements.txt`        : Dependencies needed to run the project
- `output.rmd`              : R Markdown file for generating the project report
- `output.pdf`              : Automated PDF report generated from the analysis
- `database_schema.png`     : Designed schema of database
- `documentation.pdf`       : Project details, technologies used, and instructions for running the project     


## Database Design

The database schema was designed to handle the following exemplary entities:

- **Staff**: Tracks employee details, roles, and work history.
- **Equipment**: Logs workshop tools, maintenance, and availability.
- **Vehicles**: Contains vehicle details and ownership information.
- **Transactions**: Records sales, repairs, parts orders, and other financial transactions.
- **Customers**: Manages customer information, service history, and preferences.

The schema ensures data consistency and supports future scalability for additional features.

## Data Generation

To make the system realistic and fully functional, the `fill_database.py` script automates the process of generating and populating the database with data. This includes:

- Randomized customer and vehicle information.
- Repair and transaction histories.
- Employee schedules and workshop activity logs.

The script makes use of Python’s data manipulation libraries to ensure the data is comprehensive and realistic.

## Data Analysis

The analysis section focuses on identifying trends in:

- **Repair frequency** – What types of services are most common?
- **Profitable sales** – Which services and products generate the highest profit?
- **Monthly balance** – Is the workshop profitable?


These insights are key to improving the workshop’s operational efficiency and increasing profitability.

## Report Generation

The project includes an automated report generation system. Using R Markdown (`output.rmd`), the analysis is compiled into a professional report, which is output as `output.pdf`. This report includes visualizations and summaries of key findings.

## Requirements

Ensure you have Python installed, along with any necessary packages like `pandas`, `numpy`, and `sqlalchemy`, as outlined in the `requirements.txt` file.

## Usage

1. **Configure the Project**:
   Update the `config.py` file with your database connection details.

2. **Populate the Database**:
   Run the `fill_database.py` file to populate the database with generated data

3. **Generate Reports**:
   If required, the report can be regenerated using the R markdown file (`output.rmd`).

## Contributions

- **Main Contributions**: Database design and data generation.
- **Additional Contributions**: Analysis, report generation, and documentation.


import os

def main():
    print("Select a tool to run:")
    print("1. Medication Demand Prediction")
    print("2. Advanced Demand Prediction")
    print("3. Optimized Model")
    print("4. Quick Analysis")
    print("7. Rwanda Demand Factors Analysis")
    print("8. Rwanda Demand Prediction")
    print("9. Rwanda Sales Drivers Analysis")
    print("10. Seasonal Analysis")


    choice = input("Enter the number of the tool you want to run: ")

    tools = {
        "1": "medication_demand_prediction.py",
        "2": "advanced_demand_prediction.py",
        "3": "optimized_model.py",
        "4": "quick_analysis.py",
        "7": "rwanda_demand_factors.py",
        "8": "rwanda_demand_prediction.py",
        "9": "rwanda_sales_drivers.py",
        "10": "seasonal_analysis.py",
    }

    if choice in tools:
        script = tools[choice]
        os.system(f"python {script}")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

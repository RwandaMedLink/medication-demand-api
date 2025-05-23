import os

def main():
    print("Select a tool to run:")
    print("1. Medication Demand Prediction")
    print("2. Advanced Demand Prediction")
    print("3. Optimized Model")
    print("4. Quick Analysis")
    print("5. Restock Generator")
    print("6. Restock Recommendation System")
    print("7. Rwanda Demand Factors Analysis")
    print("8. Rwanda Demand Prediction")
    print("9. Rwanda Sales Drivers Analysis")
    print("10. Seasonal Analysis")
    print("11. Simple Restock Calculator")

    choice = input("Enter the number of the tool you want to run: ")

    tools = {
        "1": "medication_demand_prediction.py",
        "2": "advanced_demand_prediction.py",
        "3": "optimized_model.py",
        "4": "quick_analysis.py",
        "5": "restock_generator.py",
        "6": "restock_recommendation_system.py",
        "7": "rwanda_demand_factors.py",
        "8": "rwanda_demand_prediction.py",
        "9": "rwanda_sales_drivers.py",
        "10": "seasonal_analysis.py",
        "11": "simple_restock.py",
    }

    if choice in tools:
        script = tools[choice]
        os.system(f"python {script}")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

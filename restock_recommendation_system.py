import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import os
from tabulate import tabulate
import sys

class MedicationRestockSystem:
    def __init__(self, model_path=None, data_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.df = None
        self.restock_recommendations = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists('reports'):
            os.makedirs('reports')
    
    def load_model(self, model_path=None):
        """Load a trained model from disk."""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            # Try to find a model in the models directory
            if os.path.exists('models'):
                model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
                if model_files:
                    self.model_path = os.path.join('models', model_files[0])
                    print(f"Auto-detected model: {self.model_path}")
        
        if not self.model_path or not os.path.exists(self.model_path):
            print("Error: No valid model path provided.")
            return False
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_data(self, data_path=None):
        """Load the dataset from disk."""
        if data_path:
            self.data_path = data_path
        
        if not self.data_path or not os.path.exists(self.data_path):
            print("Error: No valid data path provided.")
            return False
        
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Convert date columns to datetime
            date_cols = ['Date', 'sale_timestamp', 'stock_entry_timestamp', 'expiration_date']
            for col in date_cols:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                    except:
                        pass
            
            print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def generate_restock_recommendations(self, days_to_predict=30, safety_stock_days=7):
        """Generate restock recommendations based on predicted demand."""
        if self.model is None:
            print("Error: Model not loaded. Please load a model first.")
            return False
        
        if self.df is None:
            print("Error: Data not loaded. Please load data first.")
            return False
        
        print(f"Generating restock recommendations for the next {days_to_predict} days...")
        
        # Get the latest date in the dataset
        latest_date = self.df['Date'].max()
        print(f"Latest date in dataset: {latest_date.strftime('%Y-%m-%d')}")
        
        # Filter to the latest data for each product/location
        latest_data = self.df[self.df['Date'] == latest_date].copy()
        
        # Generate future dates
        future_dates = [latest_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        # Extract date features (if needed by the model)
        date_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend', 'Quarter', 'DayOfYear']
        has_date_features = all(feature in self.df.columns for feature in date_features)
        
        # Prepare storage for predictions
        all_predictions = []
        
        # Process each product/location combination
        product_locations = latest_data[['Drug_ID', 'Health_Center']].drop_duplicates()
        total_combinations = len(product_locations)
        
        print(f"Predicting demand for {total_combinations} product-location combinations...")
        
        for idx, (_, row) in enumerate(product_locations.iterrows(), 1):
            if idx % 50 == 0 or idx == total_combinations:
                print(f"Processing {idx}/{total_combinations} combinations...")
            
            drug_id = row['Drug_ID']
            health_center = row['Health_Center']
            
            # Get the base data for this product/location
            base_data = latest_data[(latest_data['Drug_ID'] == drug_id) & 
                                  (latest_data['Health_Center'] == health_center)].iloc[0].copy()
            
            # Make predictions for each future date
            for future_date in future_dates:
                # Create a new row for prediction
                predict_row = base_data.copy()
                predict_row['Date'] = future_date
                
                # Update date features if they exist in the model
                if has_date_features:
                    predict_row['Year'] = future_date.year
                    predict_row['Month'] = future_date.month
                    predict_row['Day'] = future_date.day
                    predict_row['DayOfWeek'] = future_date.weekday()
                    predict_row['IsWeekend'] = 1 if future_date.weekday() >= 5 else 0
                    predict_row['Quarter'] = (future_date.month - 1) // 3 + 1
                    predict_row['DayOfYear'] = future_date.timetuple().tm_yday
                
                # Update season if it exists
                if 'Season' in predict_row:
                    month = future_date.month
                    if month in [3, 4, 5]:
                        predict_row['Season'] = "Itumba"      # Long rainy
                    elif month in [6, 7, 8]:
                        predict_row['Season'] = "Icyi"        # Long dry
                    elif month in [9, 10, 11]:
                        predict_row['Season'] = "Umuhindo"    # Short rainy
                    else:
                        predict_row['Season'] = "Urugaryi"    # Short dry (Dec–Feb)
                
                # Update days-based features
                if 'Days_Until_Expiry' in predict_row and 'expiration_date' in predict_row:
                    predict_row['Days_Until_Expiry'] = (predict_row['expiration_date'] - future_date).days
                
                if 'Days_Since_Stock_Entry' in predict_row and 'stock_entry_timestamp' in predict_row:
                    predict_row['Days_Since_Stock_Entry'] = (future_date - predict_row['stock_entry_timestamp']).days
                
                # Convert to DataFrame
                predict_df = pd.DataFrame([predict_row])
                
                # Prepare for prediction (drop date columns and target)
                drop_cols = ['units_sold'] + [col for col in predict_df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
                X_predict = predict_df.drop(columns=drop_cols, errors='ignore')
                
                try:
                    # Make prediction
                    prediction = max(0, self.model.predict(X_predict)[0])
                    
                    # Store prediction results
                    all_predictions.append({
                        'Date': future_date,
                        'Drug_ID': drug_id,
                        'Health_Center': health_center,
                        'Province': base_data.get('Province', 'Unknown'),
                        'Predicted_Units': round(prediction),
                        'Current_Stock': base_data['available_stock'],
                        'Price_Per_Unit': base_data['Price_Per_Unit']
                    })
                except Exception as e:
                    print(f"Error predicting for {drug_id} at {health_center}: {str(e)}")
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        
        # Calculate aggregate predictions by drug and location
        group_cols = ['Drug_ID', 'Health_Center', 'Province']
        agg_predictions = predictions_df.groupby(group_cols).agg({
            'Predicted_Units': 'sum',
            'Current_Stock': 'first',
            'Price_Per_Unit': 'first'
        }).reset_index()
        
        # Rename columns for clarity
        agg_predictions.rename(columns={
            'Predicted_Units': f'Predicted_Demand_{days_to_predict}_Days'
        }, inplace=True)
        
        # Calculate daily demand rate
        agg_predictions['Daily_Demand_Rate'] = agg_predictions[f'Predicted_Demand_{days_to_predict}_Days'] / days_to_predict
        
        # Calculate safety stock (e.g., 7 days of demand)
        agg_predictions['Safety_Stock'] = agg_predictions['Daily_Demand_Rate'] * safety_stock_days
        
        # Calculate restock amount (projected demand + safety stock - current stock)
        agg_predictions['Recommended_Restock'] = (
            agg_predictions[f'Predicted_Demand_{days_to_predict}_Days'] + 
            agg_predictions['Safety_Stock'] - 
            agg_predictions['Current_Stock']
        )
        
        # Ensure no negative restock recommendations
        agg_predictions['Recommended_Restock'] = agg_predictions['Recommended_Restock'].apply(lambda x: max(0, round(x)))
        
        # Calculate days of inventory left
        agg_predictions['Days_Of_Stock_Remaining'] = (
            agg_predictions['Current_Stock'] / agg_predictions['Daily_Demand_Rate']
        ).fillna(0)
        
        # Calculate approximate cost of restock
        agg_predictions['Restock_Cost'] = agg_predictions['Recommended_Restock'] * agg_predictions['Price_Per_Unit']
        
        # Assign priority levels
        def get_priority(days_remaining):
            if days_remaining < 7:
                return "URGENT"
            elif days_remaining < 14:
                return "HIGH"
            elif days_remaining < 30:
                return "MEDIUM"
            else:
                return "LOW"
        
        agg_predictions['Restock_Priority'] = agg_predictions['Days_Of_Stock_Remaining'].apply(get_priority)
        
        # Sort by priority and restock amount
        self.restock_recommendations = agg_predictions.sort_values(
            by=['Restock_Priority', 'Recommended_Restock'], 
            ascending=[True, False]
        )
        
        # Save the recommendations
        self.restock_recommendations.to_csv('reports/restock_recommendations.csv', index=False)
        
        print(f"Restock recommendations generated and saved to 'reports/restock_recommendations.csv'")
        return True
    
    def generate_html_report(self):
        """Generate an interactive HTML report with the restock recommendations."""
        if self.restock_recommendations is None:
            print("Error: No restock recommendations available.")
            return False
        
        # Create a copy of the recommendations
        report_data = self.restock_recommendations.copy()
        
        # Format currency values
        report_data['Restock_Cost'] = report_data['Restock_Cost'].apply(lambda x: f"${x:.2f}")
        
        # Format days of stock remaining
        report_data['Days_Of_Stock_Remaining'] = report_data['Days_Of_Stock_Remaining'].apply(lambda x: f"{x:.1f}")
        
        # Add color-coding for priority
        priority_colors = {
            'URGENT': 'red',
            'HIGH': 'orange',
            'MEDIUM': 'yellow',
            'LOW': 'green'
        }
        
        # Create basic HTML for the report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medication Restock Recommendations</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                .report-date { color: #7f8c8d; margin-bottom: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th { background-color: #3498db; color: white; text-align: left; padding: 12px; }
                td { border: 1px solid #ddd; padding: 8px; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                tr:hover { background-color: #ddd; }
                .urgent { background-color: #ffcccc; }
                .high { background-color: #ffe0b3; }
                .medium { background-color: #ffffcc; }
                .low { background-color: #d9f2d9; }
                .filter-container { margin: 20px 0; }
                .chart-container { width: 100%; height: 400px; margin: 40px 0; }
                .summary-container { 
                    display: flex; 
                    flex-wrap: wrap; 
                    justify-content: space-between; 
                    margin-bottom: 30px; 
                }
                .summary-box {
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    padding: 15px;
                    margin: 10px 0;
                    width: 22%;
                    text-align: center;
                }
                .summary-number {
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                function filterTable() {
                    var input = document.getElementById('searchInput');
                    var filter = input.value.toUpperCase();
                    var table = document.getElementById('restockTable');
                    var tr = table.getElementsByTagName('tr');
                    
                    for (var i = 1; i < tr.length; i++) {
                        var td = tr[i].getElementsByTagName('td');
                        var found = false;
                        
                        for (var j = 0; j < td.length; j++) {
                            if (td[j]) {
                                var txtValue = td[j].textContent || td[j].innerText;
                                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                                    found = true;
                                    break;
                                }
                            }
                        }
                        
                        if (found) {
                            tr[i].style.display = '';
                        } else {
                            tr[i].style.display = 'none';
                        }
                    }
                }
                
                function filterByPriority(priority) {
                    var table = document.getElementById('restockTable');
                    var tr = table.getElementsByTagName('tr');
                    
                    for (var i = 1; i < tr.length; i++) {
                        var td = tr[i].getElementsByTagName('td')[8]; // Priority column
                        
                        if (priority === 'ALL') {
                            tr[i].style.display = '';
                        } else if (td) {
                            var txtValue = td.textContent || td.innerText;
                            if (txtValue === priority) {
                                tr[i].style.display = '';
                            } else {
                                tr[i].style.display = 'none';
                            }
                        }
                    }
                }
            </script>
        </head>
        <body>
            <h1>Medication Restock Recommendations</h1>
            <div class="report-date">Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</div>
            
            <div class="summary-container">
                <div class="summary-box">
                    <h3>Total Products</h3>
                    <div class="summary-number">""" + str(len(report_data)) + """</div>
                </div>
                <div class="summary-box">
                    <h3>Urgent Products</h3>
                    <div class="summary-number">""" + str(len(report_data[report_data['Restock_Priority'] == 'URGENT'])) + """</div>
                </div>
                <div class="summary-box">
                    <h3>Total Restock Units</h3>
                    <div class="summary-number">""" + f"{report_data['Recommended_Restock'].sum():,}" + """</div>
                </div>
                <div class="summary-box">
                    <h3>Estimated Cost</h3>
                    <div class="summary-number">$""" + f"{report_data['Restock_Cost'].str.replace('$', '').astype(float).sum():,.2f}" + """</div>
                </div>
            </div>
            
            <div class="chart-container">
                <canvas id="priorityChart"></canvas>
            </div>
            
            <div class="filter-container">
                <input type="text" id="searchInput" onkeyup="filterTable()" placeholder="Search for drug, health center...">
                <button onclick="filterByPriority('ALL')">All</button>
                <button onclick="filterByPriority('URGENT')">Urgent</button>
                <button onclick="filterByPriority('HIGH')">High</button>
                <button onclick="filterByPriority('MEDIUM')">Medium</button>
                <button onclick="filterByPriority('LOW')">Low</button>
            </div>
            
            <table id="restockTable">
                <tr>
                    <th>Drug ID</th>
                    <th>Health Center</th>
                    <th>Province</th>
                    <th>Current Stock</th>
                    <th>Predicted Demand</th>
                    <th>Days Remaining</th>
                    <th>Recommended Restock</th>
                    <th>Estimated Cost</th>
                    <th>Priority</th>
                </tr>
        """
        
        # Add rows to the table
        for _, row in report_data.iterrows():
            priority_class = row['Restock_Priority'].lower()
            
            html_content += f"""
                <tr class="{priority_class}">
                    <td>{row['Drug_ID']}</td>
                    <td>{row['Health_Center']}</td>
                    <td>{row['Province']}</td>
                    <td>{int(row['Current_Stock'])}</td>
                    <td>{int(row['Predicted_Demand_30_Days'])}</td>
                    <td>{row['Days_Of_Stock_Remaining']}</td>
                    <td>{int(row['Recommended_Restock'])}</td>
                    <td>{row['Restock_Cost']}</td>
                    <td>{row['Restock_Priority']}</td>
                </tr>
            """
        
        # Add chart data
        priority_counts = report_data['Restock_Priority'].value_counts()
        
        chart_data = "{"
        for priority in ['URGENT', 'HIGH', 'MEDIUM', 'LOW']:
            if priority in priority_counts:
                chart_data += f'"{priority}": {priority_counts[priority]},'
        chart_data = chart_data.rstrip(',') + "}"
        
        html_content += """
            </table>
            
            <script>
                // Create priority distribution chart
                var ctx = document.getElementById('priorityChart').getContext('2d');
                var priorityData = """ + chart_data + """;
                
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(priorityData),
                        datasets: [{
                            label: 'Number of Products by Restock Priority',
                            data: Object.values(priorityData),
                            backgroundColor: [
                                'rgba(255, 99, 132, 0.7)',
                                'rgba(255, 159, 64, 0.7)',
                                'rgba(255, 205, 86, 0.7)',
                                'rgba(75, 192, 192, 0.7)'
                            ],
                            borderColor: [
                                'rgb(255, 99, 132)',
                                'rgb(255, 159, 64)',
                                'rgb(255, 205, 86)',
                                'rgb(75, 192, 192)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Products by Restock Priority',
                                font: {
                                    size: 16
                                }
                            }
                        }
                    }
                });
            </script>
        </body>
        </html>
        """
        
        # Save the HTML report
        with open('reports/restock_report.html', 'w') as f:
            f.write(html_content)
        
        print("Interactive HTML report generated and saved to 'reports/restock_report.html'")
        return True
    
    def print_report(self):
        """Print a summary report to the console."""
        if self.restock_recommendations is None:
            print("Error: No restock recommendations available.")
            return False
        
        # Create a copy of the recommendations
        report_data = self.restock_recommendations.copy()
        
        # Group by priority for summary
        priority_summary = report_data.groupby('Restock_Priority').agg({
            'Drug_ID': 'count',
            'Recommended_Restock': 'sum',
            'Restock_Cost': 'sum'
        }).reset_index()
        
        # Calculate totals
        total_products = len(report_data)
        total_units = report_data['Recommended_Restock'].sum()
        total_cost = report_data['Restock_Cost'].sum()
        
        # Order priorities correctly
        priority_order = {'URGENT': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        priority_summary['Order'] = priority_summary['Restock_Priority'].map(priority_order)
        priority_summary = priority_summary.sort_values('Order').drop(columns=['Order'])
        
        # Print summary report
        print("\n" + "=" * 80)
        print("MEDICATION RESTOCK RECOMMENDATIONS SUMMARY")
        print("=" * 80)
        print(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Products: {total_products}")
        print(f"Total Restock Units: {total_units:,}")
        print(f"Total Estimated Cost: ${total_cost:,.2f}")
        print("\nRESTOCK PRIORITY BREAKDOWN:")
        
        # Print priority table
        priority_table = []
        for _, row in priority_summary.iterrows():
            priority_table.append([
                row['Restock_Priority'],
                row['Drug_ID'],
                f"{row['Recommended_Restock']:,}",
                f"${row['Restock_Cost']:,.2f}",
                f"{(row['Drug_ID'] / total_products) * 100:.1f}%"
            ])
        
        print(tabulate(priority_table, 
                     headers=['Priority', 'Products', 'Units', 'Cost', '% of Total'],
                     tablefmt='grid'))
        
        # Print top urgent items
        urgent_items = report_data[report_data['Restock_Priority'] == 'URGENT'].head(10)
        
        if not urgent_items.empty:
            print("\nTOP URGENT ITEMS:")
            urgent_table = []
            for _, row in urgent_items.iterrows():
                urgent_table.append([
                    row['Drug_ID'],
                    row['Health_Center'],
                    int(row['Current_Stock']),
                    int(row['Predicted_Demand_30_Days']),
                    f"{float(row['Days_Of_Stock_Remaining']):.1f}",
                    int(row['Recommended_Restock']),
                    f"${row['Restock_Cost']:,.2f}"
                ])
            
            print(tabulate(urgent_table,
                         headers=['Drug ID', 'Health Center', 'Current Stock', 'Predicted Demand', 
                                'Days Left', 'Restock Units', 'Cost'],
                         tablefmt='grid'))
        
        print("\nReports saved to 'reports' directory:")
        print("- reports/restock_recommendations.csv")
        print("- reports/restock_report.html")
        print("=" * 80)
        
        return True

def main():
    print("\n" + "=" * 80)
    print("RWANDA MEDLINK RESTOCK RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Check command line arguments
    model_path = None
    data_path = None
    days_to_predict = 30
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    if len(sys.argv) > 3:
        try:
            days_to_predict = int(sys.argv[3])
        except:
            pass
    
    # Default paths if not provided
    if not data_path:
        data_path = 'synthetic_pharma_sales.csv'
    
    # Create the restock system
    restock_system = MedicationRestockSystem(model_path, data_path)
    
    # Load the model
    if not restock_system.load_model():
        # If no specific model provided, try to search in the 'models' directory
        if os.path.exists('models'):
            model_files = [os.path.join('models', f) for f in os.listdir('models') if f.endswith('.pkl')]
            if model_files:
                print(f"Trying alternative model: {model_files[0]}")
                if not restock_system.load_model(model_files[0]):
                    print("Could not load any model. Exiting.")
                    return
            else:
                print("No models found in 'models' directory. Exiting.")
                return
        else:
            print("No 'models' directory found. Exiting.")
            return
    
    # Load the data
    if not restock_system.load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Generate restock recommendations
    if not restock_system.generate_restock_recommendations(days_to_predict=days_to_predict):
        print("Failed to generate restock recommendations. Exiting.")
        return
    
    # Generate HTML report
    restock_system.generate_html_report()
    
    # Print report summary
    restock_system.print_report()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

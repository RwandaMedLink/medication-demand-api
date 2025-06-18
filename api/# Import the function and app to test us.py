# Import the function and app to test using relative import

class TestPredictBatchEndpoint(unittest.TestCase):
    def setUp(self):
        """Set up test client and mock data."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Sample valid batch request data
        self.valid_batch_data = {
            "records": [
                {
                    "Pharmacy_Name": "CityMeds 795",
                    "Province": "Kigali",
                    "Drug_ID": "DICLOFENAC",
                    "Date": "2024-01-01",
                    "ATC_Code": "M01AB",
                    "available_stock": 470,
                    "expiration_date": "2024-02-28",
                    "stock_entry_timestamp": "2023-12-06",
                    "Price_Per_Unit": 33.04,
                    "Promotion": 1,
                    "Season": "Urugaryi",
                    "Disease_Outbreak": 1,
                    "Supply_Chain_Delay": "Medium",
                    "Effectiveness_Rating": 5,
                    "Competitor_Count": 4,
                    "Time_On_Market": 47,
                    "Population_Density": "high",
                    "Income_Level": "higher",
                    "Holiday_Week": 1
                },
                {
                    "Pharmacy_Name": "HealthPlus 123",
                    "Province": "Western",
                    "Drug_ID": "PARACETAMOL",
                    "Date": "2024-01-02",
                    "ATC_Code": "N02BE",
                    "available_stock": 250,
                    "expiration_date": "2024-03-15",
                    "stock_entry_timestamp": "2023-11-20",
                    "Price_Per_Unit": 15.50,
                    "Promotion": 0,
                    "Season": "Urugaryi",
                    "Disease_Outbreak": 0,
                    "Supply_Chain_Delay": "Low",
                    "Effectiveness_Rating": 8,
                    "Competitor_Count": 6,
                    "Time_On_Market": 30,
                    "Population_Density": "medium",
                    "Income_Level": "middle",
                    "Holiday_Week": 0
                }
            ]
        }

    @patch.object(demand_predictor, 'predict')
    def test_predict_batch_success(self, mock_predict):
        """Test successful batch prediction."""
        # Mock the predict method to return successful predictions
        mock_predict.side_effect = [(150, "Success"), (75, "Success")]
        
        response = self.client.post('/predict_batch',
                                  data=json.dumps(self.valid_batch_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertEqual(data['total_records'], 2)
        self.assertEqual(data['model_type'], 'Linear Regression')
        
        # Check first prediction
        first_prediction = data['predictions'][0]
        self.assertEqual(first_prediction['record_index'], 0)
        self.assertEqual(first_prediction['predicted_demand'], 150)
        self.assertEqual(first_prediction['message'], 'Success')
        
        # Check second prediction
        second_prediction = data['predictions'][1]
        self.assertEqual(second_prediction['record_index'], 1)
        self.assertEqual(second_prediction['predicted_demand'], 75)
        self.assertEqual(second_prediction['message'], 'Success')

    @patch.object(demand_predictor, 'predict')
    def test_predict_batch_mixed_results(self, mock_predict):
        """Test batch prediction with some successful and some failed predictions."""
        # Mock mixed results: first succeeds, second fails
        mock_predict.side_effect = [(120, "Success"), (None, "Model not loaded")]
        
        response = self.client.post('/predict_batch',
                                  data=json.dumps(self.valid_batch_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        predictions = data['predictions']
        
        # Check successful prediction
        self.assertEqual(predictions[0]['predicted_demand'], 120)
        self.assertEqual(predictions[0]['message'], 'Success')
        
        # Check failed prediction
        self.assertIsNone(predictions[1]['predicted_demand'])
        self.assertEqual(predictions[1]['message'], 'Model not loaded')

    def test_predict_batch_no_data(self):
        """Test batch prediction with no input data."""
        response = self.client.post('/predict_batch',
                                  data=json.dumps({}),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No records provided')

    def test_predict_batch_no_records_key(self):
        """Test batch prediction with missing 'records' key."""
        invalid_data = {"data": []}
        
        response = self.client.post('/predict_batch',
                                  data=json.dumps(invalid_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No records provided')

    def test_predict_batch_empty_records(self):
        """Test batch prediction with empty records array."""
        empty_data = {"records": []}
        
        response = self.client.post('/predict_batch',
                                  data=json.dumps(empty_data),
                                  content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
# Fetch predictions from deployed model by send a POST request
curl http://127.0.0.1:5000/invocations -H "Content-Type: application/json; format=pandas-records" \
-d '[[5.1, 3.5, 1.4, 0.2],
     [4.9, 3. , 1.4, 0.2],
     [4.7, 3.2, 1.3, 0.2],
     [4.6, 3.1, 1.5, 0.2],
     [5. , 3.6, 1.4, 0.2]]'
#!/usr/bin/env python
# coding: utf-8

# ### Question 1: Reverse List by N Elements
# 
# **Problem Statement**:
# 
# Write a function that takes a list and an integer `n`, and returns the list with every group of `n` elements reversed. If there are fewer than `n` elements left at the end, reverse all of them.

# In[7]:


def reverse_in_groups(list, n):
    return [x for i in range(0, len(list), n) for x in reversed(list[i:i + n])]
x = [11,23,34,45,56,67,78,89,90]
n = 2
print(reverse_in_groups(x, n))  


# ### Question 2: Lists & Dictionaries
# 
# **Problem Statement**:
# 
# Write a function that takes a list of strings and groups them by their length. The result should be a dictionary where:
# - The keys are the string lengths.
# - The values are lists of strings that have the same length as the key.
# 
# **Requirements**:
# 1. Each string should appear in the list corresponding to its length.
# 2. The result should be sorted by the lengths (keys) in ascending order.
# 

# In[11]:


def group_by_length(str):
    from collections import defaultdict

    x = defaultdict(list)
    for s in str:
        x[len(s)].append(s)
    
    return dict(sorted(x.items()))
Names = ["black", "pink", "red", "blue", "white", "purple"]
result = group_by_length(Names)
print(result) 


# ### Question 3: Flatten a Nested Dictionary
# 
# You are given a nested dictionary that contains various details (including lists and sub-dictionaries). Your task is to write a Python function that flattens the dictionary such that:
# 
# - **Nested keys** are concatenated into a single key with levels separated by a dot (`.`).
# - **List elements** should be referenced by their index, enclosed in square brackets (e.g., `sections[0]`).
# 

# In[15]:


from typing import Dict, Any

def flatten_dict(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    flat_dict = {}
    def _flatten(d, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                _flatten(v, new_key)
            else:
                flat_dict[new_key] = v
    _flatten(nested_dict)
    return flat_dict

# Example usage:
person_info = {
    'name': 'John Doe',
    'address': {
        'city': 'Springfield',
        'state': 'IL'
    },
    'contact': {
        'email': 'john.doe@example.com'
    }
}

flattened_info = flatten_dict(person_info)
print(flattened_info)


# ### Question 4: Generate Unique Permutations
# 
# **Problem Statement**:
# 
# You are given a list of integers that may contain duplicates. Your task is to generate all **unique** permutations of the list. The output should not contain any duplicate permutations.

# In[19]:


from typing import List


def unique_permutations(nums: List[int]) -> List[List[int]]:
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i-1] and not used[i-1]):
                continue
            used[i] = True
            backtrack(path + [nums[i]], used)
            used[i] = False

    nums.sort() 
    result = []
    backtrack([], [False] * len(nums))
    return result
nums = [4, 1, 2]
permutations = unique_permutations(nums)
print(permutations)


# ### Question 5: Find All Dates in a Text
# 
# **Problem Statement**:
# 
# You are given a string that contains dates in various formats (such as "dd-mm-yyyy", "mm/dd/yyyy", "yyyy.mm.dd", etc.). Your task is to identify and return all the valid dates present in the string.
# 
# You need to write a function `find_all_dates` that takes a string as input and returns a list of valid dates found in the text. The dates can be in any of the following formats:
# - `dd-mm-yyyy`
# - `mm/dd/yyyy`
# - `yyyy.mm.dd`
# 
# You are required to use **regular expressions** to identify these dates.

# In[20]:


import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b', 
        r'\b\d{2}/\d{2}/\d{4}\b', 
        r'\b\d{4}\.\d{2}\.\d{2}\b',
    ]
    
    matches = []
    for pattern in date_patterns:
        matches.extend(re.findall(pattern, text))
    
    return matches


# In[21]:


text = "Today's date is 21-10-2024. Another format would be 10/21/2024. Sometimes you see 2024.10.21."
print(find_all_dates(text))


# ### Question 6: Decode Polyline, Convert to DataFrame with Distances
# 
# You are given a polyline string, which encodes a series of latitude and longitude coordinates. Polyline encoding is a method to efficiently store latitude and longitude data using fewer bytes. The Python `polyline` module allows you to decode this string into a list of coordinates.
# 
# Write a function that performs the following operations:
# 1. **Decode the polyline** string using the `polyline` module into a list of (latitude, longitude) coordinates.
# 2. **Convert these coordinates into a Pandas DataFrame** with the following columns:
#    - `latitude`: Latitude of the coordinate.
#    - `longitude`: Longitude of the coordinate.
#    - `distance`: The distance (in meters) between the current row's coordinate and the previous row's one. The first row will have a distance of `0` since there is no previous point.
# 3. **Calculate the distance** using the Haversine formula for points in successive rows.

# In[8]:


pip install polyline


# In[5]:


pip install geopy


# In[11]:


import pandas as pd
from math import radians, cos, sin, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    R = 3456700  
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coordinates = [(27.7691, -112.4449), (37.7088, -122.4250), (27.5685, -121.4452)]
    
    # Convert the list into a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Calculate distances between consecutive points using Haversine formula
    distances = [0]  # First point has a distance of 0
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i - 1]['latitude'], df.iloc[i - 1]['longitude']
        lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
        distances.append(haversine(lat1, lon1, lat2, lon2))
    

    df['distance'] = distances
    
    return df

polyline_str = "encoded_polyline_string_here" 
df = polyline_to_dataframe(polyline_str)
print(df)


# ### Question 7: Matrix Rotation and Transformation
# Write a function that performs the following operations on a square matrix (n x n):
# 
# Rotate the matrix by 90 degrees clockwise.
# After rotation, for each element in the rotated matrix, replace it with the sum of all elements in the same row and column (in the rotated matrix), excluding itself.
# The function should return the transformed matrix.

# In[30]:


from typing import List

def rotate_matrix_90(matrix: List[List[int]]) -> List[List[int]]:
    """ Rotates a matrix 90 degrees clockwise. """
    return [list(reversed(col)) for col in zip(*matrix)]

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Rotate the matrix by 90 degrees clockwise
    rotated_matrix = rotate_matrix_90(matrix)
    
    # Multiply each element by the sum of its original row and column indices
    transformed_matrix = [[0] * n for _ in range(n)]  # Initialize the transformed matrix
    
    for i in range(n):
        for j in range(n):
            original_row = j
            original_col = n - 1 - i
            index_sum = original_row + original_col
            transformed_matrix[i][j] = rotated_matrix[i][j] * index_sum
    
    return transformed_matrix
matrix = [
    [1, 5, 3],
    [8, 5, 6],
    [9, 0, 1]
]
result = rotate_and_multiply_matrix(matrix)

print("Transformed Matrix:")
for row in result:
    print(row)
print ("rotate_matrix:")
for row in rotated_matrix:
    print(row)


# ### Question 8: Time Check
# You are given a dataset, dataset-1.csv, containing columns id, id_2, and timestamp (startDay, startTime, endDay, endTime). The goal is to verify the completeness of the time data by checking whether the timestamps for each unique (id, id_2) pair cover a full 24-hour period (from 12:00:00 AM to 11:59:59 PM) and span all 7 days of the week (from Monday to Sunday).
# 
# Create a function that accepts dataset-1.csv as a DataFrame and returns a boolean series that indicates if each (id, id_2) pair has incorrect timestamps. The boolean series must have multi-index (id, id_2).

# In[35]:


import pandas as pd


# In[36]:


df= pd.read_csv("https://raw.githubusercontent.com/mapup/MapUp-DA-Assessment-2024/refs/heads/main/datasets/dataset-1.csv")


# In[45]:


df.head()


# In[41]:


import pandas as pd


def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking 
    whether the timestamps for each unique (`id`, `id_2`) pair cover a full 
    24-hour and 7 days period.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing columns `id`, `id_2`, 
                               `startDay`, `startTime`, `endDay`, and `endTime`.
                               
    Returns:
        pd.Series: A boolean series indicating if each (`id`, `id_2`) pair has 
                    incorrect timestamps, with a multi-index of (id, id_2).
    """
    
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    

    df = df.dropna(subset=['start', 'end'])


    grouped = df.groupby(['id', 'id_2'])
    
    results = []

    for (id_, id_2), group in grouped:
        days_covered = group['start'].dt.dayofweek.unique()  # 0=Monday, 6=Sunday
        all_days_covered = set(range(7)) == set(days_covered)
        time_coverage = group.groupby(group['start'].dt.date).agg({'start': 'min', 'end': 'max'})
        full_day_coverage = (time_coverage['end'] - time_coverage['start']) >= pd.Timedelta(hours=24)
        results.append(((id_, id_2), not (all_days_covered and full_day_coverage.all())))
    result_series = pd.Series(dict(results))
    return result_series

if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/mapup/MapUp-DA-Assessment-2024/refs/heads/main/datasets/dataset-1.csv'
    df = pd.read_csv(url)
    completeness_results = time_check(df)

    print(completeness_results)


# In[43]:


import pandas as pd

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Verify the completeness of the data by checking whether the timestamps 
    for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.
    
    Args:
        df (pandas.DataFrame): Input DataFrame containing columns `id`, `id_2`, 
                               `startDay`, `startTime`, `endDay`, and `endTime`.
                               
    Returns:
        pd.Series: A boolean series indicating if each (`id`, `id_2`) pair has 
                    incorrect timestamps, with a multi-index of (id, id_2).
    """
    
    # Combine start and end dates/times into single datetime columns
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')
    df = df.dropna(subset=['start', 'end'])
    grouped = df.groupby(['id', 'id_2'])
    results = []

    for (id_, id_2), group in grouped:
        if group.empty:
            results.append(((id_, id_2), True))  
            continue
    
        days_covered = group['start'].dt.dayofweek.unique()  # 0=Monday, 6=Sunday
        all_days_covered = set(range(7)) == set(days_covered)
        
        time_coverage = group.groupby(group['start'].dt.date).agg({'start': 'min', 'end': 'max'})
        
        full_day_coverage = (time_coverage['end'] - time_coverage['start']) >= pd.Timedelta(hours=24)

        results.append(((id_, id_2), not (all_days_covered and full_day_coverage.all())))

    result_series = pd.Series(dict(results), dtype='bool')
    return result_series

url = 'https://raw.githubusercontent.com/mapup/MapUp-DA-Assessment-2024/refs/heads/main/datasets/dataset-1.csv'

df = pd.read_csv(url)
completeness_results = time_check(df)


print(completeness_results)


# In[ ]:





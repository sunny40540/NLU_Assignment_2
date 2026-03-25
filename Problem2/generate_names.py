# Generates a dataset of 1000 diverse Indian names for RNN training
import os
import random

INDIAN_NAMES = [
    # --- North Indian / Hindi Names (Male) ---
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayaan", "Krishna",
    "Ishaan", "Sai", "Arnav", "Shaurya", "Atharv", "Advait", "Rudra", "Dhruv",
    "Kabir", "Ansh", "Manav", "Ritvik", "Rohan", "Mohit", "Nikhil", "Rahul",
    "Aman", "Vikas", "Suresh", "Ramesh", "Rajesh", "Dinesh", "Mukesh", "Rakesh",
    "Pradeep", "Sandeep", "Deepak", "Vinay", "Ajay", "Vijay", "Sanjay", "Neeraj",
    "Pankaj", "Manoj", "Sunil", "Anil", "Vikram", "Gaurav", "Naveen", "Ashish",
    "Hemant", "Sumit", "Amit", "Ankit", "Varun", "Karan", "Akash", "Harsh",
    "Yash", "Parth", "Dev", "Ravi", "Kapil", "Sachin", "Tushar", "Kunal",

    # --- North Indian / Hindi Names (Female) ---
    "Ananya", "Aanya", "Aadhya", "Saanvi", "Myra", "Ira", "Aisha", "Diya",
    "Priya", "Neha", "Pooja", "Ritu", "Nisha", "Sunita", "Meena", "Kavita",
    "Rekha", "Geeta", "Sonia", "Seema", "Anjali", "Swati", "Preeti", "Jyoti",
    "Kiran", "Rani", "Radha", "Sapna", "Asha", "Usha", "Poonam", "Komal",
    "Shalini", "Rashmi", "Shikha", "Pallavi", "Manisha", "Divya", "Shruti", "Tanvi",
    "Riya", "Tanya", "Sneha", "Megha", "Ritika", "Nikita", "Shweta", "Bhavna",
    "Vidya", "Naina", "Kashish", "Zara", "Kiara", "Avani", "Srishti", "Kriti",
    "Garima", "Anamika", "Chandni", "Payal", "Kajal", "Simran", "Sakshi", "Ishita",

    # --- South Indian Names (Male) ---
    "Surya", "Karthik", "Srinivas", "Venkatesh", "Prasad", "Nagaraj", "Ganesh",
    "Harish", "Suresh", "Mahesh", "Ramesh", "Lokesh", "Rajkumar", "Prabhu",
    "Arun", "Varun", "Sathish", "Senthil", "Murugan", "Balaji", "Rajan",
    "Shankar", "Kumar", "Anand", "Mohan", "Gopal", "Hari", "Siva", "Murali",
    "Vijayan", "Karthikeyan", "Saravanan", "Mani", "Raja", "Selvam", "Pandian",
    "Kannan", "Nathan", "Srikanth", "Nandakumar", "Raghav", "Pranav", "Tarun",
    "Bharath", "Charan", "Dhruvan", "Eshan", "Farhan", "Girish", "Hitesh",

    # --- South Indian Names (Female) ---
    "Lakshmi", "Saraswati", "Meenakshi", "Deepika", "Padma", "Kalyani", "Sowmya",
    "Supriya", "Harini", "Madhavi", "Revathi", "Vasudha", "Sahana", "Varsha",
    "Keerthi", "Deepthi", "Swathi", "Sudha", "Bhanu", "Chitra", "Lavanya",
    "Amrutha", "Bhargavi", "Chaitra", "Dhanya", "Gayathri", "Hamsini", "Janani",
    "Kavitha", "Lekha", "Mythili", "Nandini", "Oviya", "Pavithra", "Ramya",
    "Sandhya", "Tejaswini", "Urmila", "Vaishali", "Yamini", "Ashwini", "Aparna",

    # --- Bengali Names ---
    "Sourav", "Ayan", "Arnab", "Subham", "Debjit", "Rik", "Soham", "Debayan",
    "Aritra", "Sayan", "Aniket", "Sabyasachi", "Partha", "Anirban", "Dipankar",
    "Swastika", "Trisha", "Aditi", "Moumita", "Anindita", "Sayantani", "Rituparna",
    "Paromita", "Sudipta", "Mithun", "Kaushik", "Chiranjit", "Tapas", "Bhaskar",
    "Debashis", "Jayanta", "Mrinal", "Prosenjit", "Rajdeep", "Saugata", "Tamal",
    "Bidisha", "Chandrima", "Devlina", "Esha", "Gargi", "Haimanti", "Ipsita",

    # --- Marathi / Gujarati Names ---
    "Omkar", "Tanmay", "Siddharth", "Shripad", "Chinmay", "Aniruddha", "Tejas",
    "Pratik", "Hrithik", "Niranjan", "Unmesh", "Yogesh", "Apurva", "Ketaki",
    "Gauri", "Mugdha", "Prajakta", "Renuka", "Smita", "Vrinda", "Manasi",
    "Chaitali", "Jayesh", "Mihir", "Nishant", "Parimal", "Rutvik", "Sagar",
    "Tushal", "Utkarsh", "Vipul", "Yatin", "Alpesh", "Bhavin", "Chirag",
    "Dhawal", "Falgun", "Gautam", "Hiren", "Jignesh", "Kishan", "Lalit",

    # --- Punjabi / Sikh Names ---
    "Gurpreet", "Harpreet", "Manpreet", "Navjot", "Jaspreet", "Amandeep",
    "Sukhwinder", "Kuldeep", "Hardeep", "Iqbal", "Jasbir", "Kartar",
    "Lakhwinder", "Maninder", "Narinder", "Parminder", "Ranjit", "Satnam",
    "Tejpal", "Udham", "Balvinder", "Charanjit", "Dalvinder", "Ekam",
    "Fateh", "Gurbani", "Harmeet", "Inderpal", "Jasleen", "Kamalpreet",
    "Lovleen", "Manmeet", "Navneet", "Onkar", "Paramjit", "Ravneet",

    # --- Additional Common Indian Names ---
    "Abhishek", "Akshay", "Alok", "Amitabh", "Ashok", "Bharat", "Chandan",
    "Daksh", "Ekansh", "Faiz", "Govind", "Himanshu", "Ishan", "Jagdish",
    "Kartik", "Lakshman", "Madhav", "Narayan", "Om", "Pawan", "Raghunath",
    "Shivam", "Trilok", "Umang", "Vansh", "Wriddhiman", "Yuvraj", "Zaheer",
    "Abhinav", "Bhuvan", "Chiranjeevi", "Devendra", "Eknath", "Firoz",
    "Girdhar", "Harshad", "Indrajit", "Jagannath", "Kaustubh", "Likhith",
    "Madhur", "Narendra", "Ojas", "Praneet", "Qadir", "Rajendra", "Shashank",
    "Tarang", "Uddhav", "Vineet", "Yagnesh", "Zubin", "Aakash", "Badrinath",
    "Chandrashekar", "Dharmanand", "Ekalavya", "Falak", "Govardhan", "Harshit",

    # --- Additional Female Indian Names ---
    "Arundhati", "Bhairavi", "Chandra", "Devika", "Ekta", "Falguni", "Ganga",
    "Hemlata", "Indira", "Jayashree", "Kamala", "Lalita", "Maitri", "Namrata",
    "Ojasvi", "Padmini", "Ragini", "Saroj", "Tara", "Uma", "Vasundhara",
    "Yamuna", "Zarina", "Ahilya", "Bhavani", "Charulata", "Damini", "Ela",
    "Hema", "Jaya", "Kusum", "Madhuri", "Niranjana", "Prerna", "Rukmini",
    "Savita", "Tulsi", "Ujjwala", "Vaishnavi", "Vrushali", "Anushka", "Bhoomika",
    "Chhaya", "Durga", "Fatima", "Geetika", "Hiranmayi", "Isha", "Juhi",
    "Kamini", "Leela", "Malini", "Nalini", "Parvati", "Rohini", "Sita",
    "Trishna", "Vaidehi", "Yamini", "Aaliya", "Babita", "Chanchal", "Deepshikha",
    "Eshani", "Firdaus", "Gulnaz", "Hasina", "Indu", "Jhanvi", "Khushi",
    "Lakshika", "Mohini", "Nivedita", "Pratibha", "Rajni", "Shaila", "Tanuja",
    "Uditi", "Vibha", "Wafa", "Zeenat",

    # --- More Male Names (to reach 1000) ---
    "Abhay", "Achyut", "Adarsh", "Agastya", "Akshit", "Alankrit", "Ambar",
    "Amol", "Anant", "Angad", "Animesh", "Anmol", "Anup", "Apoorv",
    "Archit", "Arihant", "Aryan", "Ashwin", "Atul", "Avnish", "Ayush",
    "Badal", "Bakul", "Balbir", "Barun", "Bhavesh", "Bhupinder", "Bikram",
    "Binod", "Bipin", "Birendra", "Braham", "Brij", "Budhil", "Chaitanya",
    "Chakradhar", "Champak", "Chandresh", "Chetan", "Chitresh", "Damodar",
    "Darshan", "Deven", "Dhiraj", "Dilip", "Dipak", "Divyanshu", "Drupad",
    "Dushyant", "Eklavya", "Feroze", "Gagan", "Gajendra", "Gaurang",
    "Ghanshyam", "Giriraj", "Gunjan", "Gurmeet", "Harsh", "Harshal",
    "Hemendra", "Jagmohan", "Jaideep", "Janak", "Jatin", "Jitender",
    "Joginder", "Jugal", "Kalyan", "Kamal", "Kanhaiya", "Kapil", "Kedar",
    "Keshav", "Kirti", "Kundan", "Lalit", "Lokendra", "Luv", "Madhukar",
    "Madan", "Mahavir", "Mahendra", "Mainak", "Malhar", "Mangal",
    "Manish", "Manohar", "Mayank", "Milan", "Milin", "Mohandas",
    "Mukund", "Nagendra", "Nakul", "Nalin", "Naman", "Naresh",
    "Navin", "Neel", "Neelesh", "Nihar", "Nikunj", "Nilesh", "Nimish",
    "Niraj", "Nitesh", "Nripendra", "Ojaswin", "Padmanabh", "Palash",
    "Pankil", "Paras", "Piyush", "Prakhar", "Pramod", "Prashanth",
    "Prateek", "Pravin", "Puneet", "Purushottam", "Radheshyam",
    "Raghavendra", "Raghu", "Rajat", "Rajeev", "Rajiv", "Ramakant",
    "Raman", "Ramgopal", "Randhir", "Ranjeet", "Ratan", "Ravindra",
    "Rishabh", "Rohit", "Rupesh", "Rustom", "Saket", "Sameer",
    "Sankalp", "Santosh", "Sarang", "Sarvesh", "Satish", "Shailendra",
    "Shakti", "Shantanu", "Sharad", "Shekar", "Shirish", "Shobit",
    "Shreyas", "Shubham", "Shyam", "Sohan", "Subodh", "Sudhir",
    "Sujit", "Sumanth", "Sundaram", "Suraj", "Surendra", "Sushil",
    "Swapnil", "Tarun", "Trilokesh", "Tribhuvan", "Tushar", "Uday",
    "Ujjwal", "Uttam", "Vaibhav", "Vallabh", "Vamsi", "Vedant",
    "Vidhya", "Vimal", "Vinod", "Viren", "Vishnu", "Vivek",
    "Yash", "Yogendra", "Yudhishthir", "Zafar", "Zuhaib",

    # --- More Female Names (to reach 1000) ---
    "Aishwarya", "Akanksha", "Amruta", "Archana", "Basanti", "Bindiya",
    "Champa", "Darpana", "Eravati", "Falgunee", "Gitanjali", "Hansika",
    "Ishwarya", "Jhalak", "Karuna", "Lajwanti", "Mandira", "Nayantara",
    "Oorja", "Pushpa", "Roopal", "Samridhi", "Trupti", "Urvi",
    "Vandana", "Yashoda", "Zarqa", "Achala", "Bharati", "Chameli",
    "Devyani", "Gagandeep", "Hemali", "Jivika", "Kalpana",
    "Lata", "Mamta", "Nandita", "Panchali", "Radhika", "Shobha",
    "Tarini", "Urvashi", "Vilasini", "Yojana", "Aanchal",
    "Beena", "Chandrika", "Dakshayini", "Elina",
]


def generate_names_file(output_path="TrainingNames.txt", n=1000):
    """
    Generate a file with 1000 Indian names for RNN training.
    
    If we have more than n unique base names, we sample n of them.
    If not, we use all and add slight variations to reach n.
    
    Args:
        output_path (str): Path to save the names file
        n (int): Number of names to generate (default: 1000)
    """
    # Remove duplicates and ensure all are proper names
    unique_names = list(set(INDIAN_NAMES))
    random.seed(42)
    random.shuffle(unique_names)

    # If we have enough names, just take the first n
    if len(unique_names) >= n:
        selected = unique_names[:n]
    else:
        # Pad with slight variations (add common suffixes)
        selected = list(unique_names)
        suffixes = ["a", "i", "an", "ya", "ika", "esh", "raj", "deep", "preet", "dev"]
        base_idx = 0
        while len(selected) < n:
            base = unique_names[base_idx % len(unique_names)]
            suffix = suffixes[base_idx % len(suffixes)]
            new_name = base + suffix
            if new_name not in selected:
                selected.append(new_name)
            base_idx += 1

    # Sort alphabetically for readability
    selected.sort()

    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for name in selected:
            f.write(name + "\n")

    print(f"[INFO] Generated {len(selected)} Indian names -> {output_path}")
    print(f"       Sample names: {selected[:10]}")
    return selected


if __name__ == "__main__":
    generate_names_file()

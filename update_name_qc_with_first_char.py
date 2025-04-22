import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
from tqdm import tqdm

# Load the SentenceTransformer model
tqdm.pandas()
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use your custom model if necessary
#df = pd.read_excel(r"C:\Users\Desk0012\Downloads\output_file5456.xlsx",sheet_name='Sheet3')

# List of last names
last_name = [
    "Kumar", "Mishra", "Sharma", "Patel", "Yadav", "Chauhan", "Gupta", "Singh", "Reddy", 
    "Nair", "Naidu", "Mehta", "Iyer", "Rao", "Rajput", "Jha", "Soni", "Bhat", "Pandey", 
    "Verma", "Desai", "Thakur", "Khatri", "Baniya", "Chand", "Choudhury", "Agarwal", "Maharaj", 
    "Bhagat", "Sahu", "Kumawat", "Lohar", "Pundir", "Tiwari", "Saini", "Bansal", "Chaudhary", 
    "Bhatt", "Khandelwal", "Jain", "Mewari", "Patnaik", "Bhardwaj", "Kapoor", "Dhawan", "Khanna", 
    "Chopra", "Ghosh", "Sen", "Bose", "Das", "Sarkar", "Rathore", "Vora", "Shukla", "Tiwari", 
    "Trivedi", "Rastogi", "Sridhar", "Vasudev", "Madhav", "Nayak", "Pawar", "Bora", "Kaur", 
    "Singh", "Khan", "Ansari", "Pathan", "Syed", "Malik", "Farooqi", "Bukhari", "Rizvi", "Zafar", 
    "Kale", "Lokhande", "Hussain", "Shaikh", "Jadhav", "Patil", "Sawant", "Deshmukh", "Babar", 
    "Shinde", "Salvi", "Padhye", "Bhatnagar", "Vaidya", "Shah", "Gandhi", "Dewan", "Vishwakarma", 
    "Uppal", "Rungta", "Awasthi", "Chandran", "Behera", "Bansal", "Khandelwal", "Sarkar", 
    "Bhargava", "Bajpai", "Chopra", "Rajeev", "Mishra", "Garg", "Patnaik", "Rathi", "Parikh", 
    "Zaveri", "Jindal", "Tiwari", "Seth", "Dhillon", "Malhotra", "Bhatia", "Garg", "Suri", 
    "Gandhi", "Agarwal", "Mishra", "Shukla", "Kohli", "Mathur", "Yadav", "Aggarwal", "Chandran", 
    "Iyer", "Shah", "Dixit", "Sharma", "Vasudev", "Bhargava", "Shastri", "Patel", "Bansal", 
    "Lodha", "Singhal", "Thapar", "Verma", "Narayan", "Pillai", "Subramanian", "Krishnan", 
    "Ravichandran", "Nair", "Shenoy", "Menon", "Pillai", "Chandra", "Anand", "Gupta", "Bhaskar", 
    "Desai", "Tiwari", "Bansal", "Hegde", "Prakash", "Kulkarni", "Rai", "Singh", "Mishra", 
    "Reddy", "Rajendran", "Kannan", "Pillai", "Kumar", "Chakraborty", "Sen", "Lal", "Pandey", 
    "Ravindran", "Venu", "Satyanarayana", "Manoharan", "Subramanian", "Ayyar", "Raghavan", 
    "Sreenivasan", "Krishna", "Venkatesh", "Ranganathan", "Vijayan", "Krishnan", "Lakshmanan", 
    "Baskar", "Perumal", "Rajagopal", "Rajasekaran", "Kailasam", "Azhagiri", "Kumarasamy", 
    "Shanmugam", "Kandasamy", "Srinivasan", "Murugan", "Sivakumar", "Ravichandran", "Ravindra","Ghoshal","gopal",
    'Acharya', 'Adhaulia', 'Adhya', 'Aenugu', 'Afreen', 'Afsar', 'Aftab', 'Agarawal', 'Agarwal', 'Agate',
    'Aggarwal', 'Aggrawal', 'Agrahari', 'Agrawal', 'Ahamad', 'Ahluwalia', 'Ahmad', 'Ahmed', 'Ahuja', 'Aishwary',
    'Akhtar', 'Akolkar', 'Alam', 'Aleem', 'Ali', 'Alwadhi', 'Amble', 'Anam', 'Anand', 'Andra',
    'Anjum', 'Anne', 'Ansari', 'Anupama', 'Anwar', 'Apte', 'Arif', 'Arora', 'Arya', 'Ashraf',
    'Asija', 'Asthana', 'Ataher', 'Athar', 'Atri', 'Attraa', 'Atwal', 'Aurora', 'Awasthi', 'Azam',
    'Azhar', 'Azmi', 'Babu', 'Badal', 'Badami', 'Baddur', 'Baghel', 'Bahl', 'Bahri', 'Bai',
    'Bail', 'Bains', 'Bais', 'Baiswar', 'Bajaj', 'Bajpai', 'Bajwa', 'Bakshi', 'Bal', 'Bala',
    'Balakrishnan', 'Balan', 'Balasubramanian', 'Balay', 'Bali', 'Baliga', 'Balkoty', 'Balmiki', 'Balot', 'Bandi',
    'Banerjee', 'Banik', 'Bano', 'Banothu', 'Bansal', 'Bapat', 'Bapna', 'Barad', 'Barai', 'Baral',
    'Baranwal', 'Baria', 'Barman', 'Barnawal', 'Barnwal', 'Basak', 'Bassi', 'Basu', 'Bath', 'Batra',
    'Batta', 'Bauri', 'Bava', 'Bawa', 'Bedi', 'Begam', 'Begum', 'Behera', 'Behl', 'Behura',
    'Behuria', 'Belide', 'Ben', 'Bera', 'Bhadauria', 'Bhadauriya', 'Bhadouriya', 'Bhagat', 'Bhakta', 'Bhal',
    'Bhalla', 'Bhandari', 'Bharadwaj', 'Bhardwaj', 'Bhargava', 'Bhasin', 'Bhaskar', 'Bhat', 'Bhatia', 'Bhatnagar',
    'Bhatt', 'Bhattacharjee', 'Bhattacharya', 'Bhattacharyya', 'Bhatti', 'Bhavsar', 'Bhorkhade', 'Bhriguvanshi', 'Bhukya', 'Bhushan',
    'Bibi', 'Bindal', 'Bir', 'Bisht', 'Bistagond', 'Biswas', 'Boase', 'Bobal', 'Bora', 'Borah',
    'Borde', 'Borra', 'Bose', 'Brahmbhatt', 'Brar', 'Buch', 'Bugalia', 'Bumb', 'Butala', 'Cauhan',
    'Chacko', 'Chad', 'Chada', 'Chadha', 'Chahal', 'Chakma', 'Chakrabarti', 'Chakrabortty', 'Chakraborty', 'Chana',
    'Chand', 'Chanda', 'Chandel', 'Chander', 'Chandra', 'Chandran', 'Char', 'Chary', 'Chatterjee', 'Chaturvedi',
    'Chaubey', 'Chaudhari', 'Chaudhary', 'Chaudhry', 'Chaudhuri', 'Chaudry', 'Chauhan', 'Chaurasia', 'Chaurasiya', 'Chavan',
    'Chawla', 'Chechani', 'Cheema', 'Chellani', 'Cherian', 'Chetan', 'Chhabra', 'Chhateja', 'Chhipa', 'Chirag',
    'Chitranshi', 'Chokshi', 'Chopra', 'Choubey', 'Choudhary', 'Choudhry', 'Choudhury', 'Chouhan', 'Chowdhury', 'Comar',
    'Contractor', 'Dâ€™Alia', 'Dada', 'Dadarwal', 'Dagade', 'Dahiya', 'Dahlan', 'Dalal', 'Dani', 'Dar',
    'Dara', 'Darshan', 'Das', 'Dasgupta', 'Dash', 'Dasila', 'Dass', 'Date', 'Datta', 'Dave',
    'Dayal', 'De', 'Debnath', 'Deep', 'Deo', 'Deol', 'Desabathula', 'Desai', 'Deshmukh', 'Deshpande',
    'Dev', 'Devan', 'Devi', 'Dewan', 'Dey', 'Dhaka', 'Dhaliwal', 'Dhamane', 'Dhar', 'Dhaulakhandi',
    'Dhawan', 'Dhillon', 'Dhiman', 'Dhindhwal', 'Dhingra', 'Dhinoja', 'Dhoni', 'Dikshit', 'Din', 'Dinu',
    'Divan', 'Dixit', 'Doctor', 'Dohrey', 'Dora', 'Doshi', 'Dua', 'Dube', 'Dubey', 'Dudeja',
    'Dugal', 'Dugar', 'Durgavansh', 'Durgavanshi', 'Dutt', 'Dutta', 'Dwevedi', 'Dwivedi', 'Dyal', 'Edwin',
    'Fathima', 'Fatima', 'Fernandes', 'Francis', 'Gaba', 'Gade', 'Gadge', 'Gaikwad', 'Gakhar', 'Gala',
    'Ganapathy', 'Gandhi', 'Ganesan', 'Ganesh', 'Ganguly', 'Gangwal', 'Gangwar', 'Ganore', 'Gara', 'Garde',
    'Garg', 'Garud', 'Gaur', 'Gautam', 'Gayakwad', 'Gayathri', 'Gehlot', 'Gera', 'Ghose', 'Ghosh',
    'Gill', 'Giri', 'Gite', 'Goda', 'Goel', 'Gokhale', 'Gola', 'Gole', 'Golla', 'Gomashe',
    'Gond', 'Gopal', 'Goswami', 'Gour', 'Govil', 'Goyal', 'Grewal', 'Grover', 'Gudepu', 'Guduri',
    'Guha', 'Gujrati', 'Gulati', 'Gupta', 'Gurjar', 'Gurnani', 'Gurubhaiye', 'Gutti', 'Halder', 'Handa',
    'Hanif', 'Hans', 'Hansdah', 'Hanul', 'Haokip', 'Hari', 'Hasija', 'Hasnain', 'Hasni', 'Hayer',
    'Hayre', 'Heblikar', 'Hegde', 'Hora', 'Hosur', 'Huque', 'Hussain', 'Inavolu', 'Indora', 'Iqbal',
    'Islam', 'Israni', 'Issac', 'Iyengar', 'Iyer', 'Jadhav', 'Jafri', 'Jagaragallu', 'Jaggi', 'Jagtap',
    'Jain', 'Jaiswal', 'Jaitley', 'Jajoo', 'Jamloki', 'Jani', 'Jassal', 'Jauhari', 'Jawed', 'Jay',
    'Jayaraman', 'Jedhe', 'Jena', 'Jeykumaran', 'Jha', 'Jhaveri', 'Jhindal', 'Jindal', 'Jivani', 'Jodha',
    'Johal', 'Johari', 'Jose', 'Joseph', 'Joshi', 'Junankar', 'Juned', 'Juneja', 'Jushantan', 'Kadakia',
    'Kade', 'Kairati', 'Kakar', 'Kala', 'Kale', 'Kalita', 'Kalla', 'Kalra', 'Kalsi', 'Kamble',
    'Kamdar', 'Kanchan', 'Kanchapu', 'Kancharla', 'Kanchhal', 'Kanda', 'Kannan', 'Kannaujia', 'Kanojia', 'Kansal',
    'Kant', 'Kapadia', 'Kapoor', 'Kapur', 'Kar', 'Kara', 'Karan', 'Kari', 'Karkhele', 'Karnik',
    'Karpe', 'Kasaudhan', 'Kasera', 'Kashyap', 'Kasniya', 'Kata', 'Katiyar', 'Katta', 'Kaul', 'Kaur',
    'Kaushik', 'Kavi', 'Keer', 'Kesarwani', 'Keshari', 'Keshri', 'Keshwani', 'Kewlani', 'Khaliq', 'Khalsa',
    'Khan', 'Khandekar', 'Khandelwal', 'Khanna', 'Khare', 'Khatoon', 'Khatri', 'Khatun', 'Khobragade', 'Khokhar',
    'Khosla', 'Khurana', 'Kibe', 'Kidwai', 'Kishor', 'Kisshan', 'Kochar', 'Kohli', 'Konda', 'Koppula',
    'Kori', 'Korpal', 'Koshy', 'Kota', 'Kothari', 'Kotturu', 'Krish', 'Krishna', 'Krishnamurthy', 'Krishnan',
    'Kukreja', 'Kulal', 'Kulkarni', 'Kulsum', 'Kumar', 'Kumari', 'Kumer', 'Kunda', 'Kurian', 'Kuruvilla',
    'Kushalappa', 'Kushwaha', 'Kwatra', 'Lad', 'Lakhmani', 'Lal', 'Lala', 'Lalchandani', 'Lall', 'Lalla',
    'Lamba', 'Lanka', 'Lata', 'Lavudya', 'Lockwani', 'Loke', 'Loyal', 'Luthra', 'Ma', 'Macharla',
    'Machhan', 'Machra', 'Madaan', 'Madala', 'Madan', 'Maddeshiya', 'Maddipatla', 'Madina', 'Magar', 'Magdum',
    'Mahajan', 'Mahal', 'Mahalka', 'Mahankali', 'Mahapatra', 'Maharaj', 'Mahato', 'Mahawar', 'Maheshwari', 'Mahmood',
    'Mahto', 'Majeed', 'Majhi', 'Majumdar', 'Makavan', 'Malekar', 'Malhotra', 'Malik', 'Mall', 'Mallick',
    'Malviya', 'Malwan', 'Mammen', 'Manchanda', 'Manchikanti', 'Mand', 'Manda', 'Mandadi', 'Mandal', 'Mander',
    'Mane', 'Mangal', 'Mangat', 'Mangla', 'Mani', 'Manjhi', 'Manju', 'Mann', 'Manna', 'Mannan',
    'Manne', 'Masood', 'Mathai', 'Mathur', 'Matthai', 'Maurya', 'Meda', 'Meena', 'Meghwal', 'Mehan',
    'Meharda', 'Mehendale', 'Mehra', 'Mehrotra', 'Mehta', 'Meka', 'Memon', 'Menon', 'Merchant', 'Mhaske',
    'Middinti', 'Minhas', 'Mishra', 'Misra', 'Mistry', 'Mital', 'Mitra', 'Mittal', 'Mitter', 'Modi',
    'Mody', 'Mohan', 'Mohandas', 'Mohanty', 'Molla', 'Mondal', 'Morar', 'More', 'Motwani', 'Mourya',
    'Mukherjee', 'Mukhopadhyay', 'Munda', 'Muni', 'Munir', 'Munshi', 'Murlidhar', 'Murmu', 'Murthy', 'Murty',
    'Mustafa', 'Mutti', 'Naaz', 'Nadaf', 'Nadhe', 'Nadig', 'Nadkarni', 'Nag', 'Nagar', 'Nagarajan',
    'Nagi', 'Nagoria', 'Nagraj', 'Nagy', 'Naidu', 'Naik', 'Nair', 'Nalla', 'Nanda', 'Nandakumar',
    'Nandanwar', 'Nanwani', 'Naqvi', 'Narain', 'Narala', 'Narang', 'Narasimhan', 'Narayan', 'Narayanan', 'Narula',
    'Nasreen', 'Natarajan', 'Nath', 'Natt', 'Navas', 'Nayak', 'Nayar', 'Nazareth', 'Negi', 'Nichit',
    'Nigam', 'Nigan', 'Nischal', 'Nisha', 'Nori', 'Oak', 'Obed', 'Ojha', 'Om', 'Omkar',
    'Oommen', 'Oswal', 'Oza', 'Padmanabhan', 'Pai', 'Paighan', 'Pal', 'Palan', 'Palata', 'Palisetti',
    'Pall', 'Palla', 'Pallav', 'Panchal', 'Pandey', 'Pandit', 'Pandya', 'Panesar', 'Panghal', 'Pant',
    'Paramar', 'Parashar', 'Pareek', 'Parekh', 'Parihar', 'Parikh', 'Parmar', 'Parmer', 'Parsa', 'Parul',
    'Parween', 'Pasupuleti', 'Paswan', 'Patal', 'Patel', 'Pathak', 'Pathan', 'Patil', 'Patla', 'Patra',
    'Pattnaik', 'Pau', 'Paul', 'Pawar', 'Pereddy', 'Peri', 'Pillai', 'Pillay', 'Pingle', 'Poddar',
    'Pokhriyal', 'Porwal', 'Prabhakar', 'Prabhu', 'Pradhan', 'Prajapati', 'Prakash', 'Prasad', 'Prasath', 'Prashad',
    'Praveenchand', 'Pravesh', 'Preetham', 'Priya', 'Priyadarshi', 'Priyadarshini', 'Pundir', 'Puri', 'Purkayastha', 'Purohit',
    'Purwar', 'Putta', 'Quadiri', 'Quraishi', 'Radhakrishnan', 'Raghavan', 'Rai', 'Raj', 'Raja', 'Rajagopal',
    'Rajagopalan', 'Rajak', 'Rajan', 'Rajendra', 'Rajesh', 'Rajput', 'Raju', 'Rakhecha', 'Ram', 'Rama',
    'Ramachandran', 'Ramakrishnan', 'Raman', 'Ramanathan', 'Ramaswamy', 'Ramesh', 'Rana', 'Randhawa', 'Ranganathan', 'Rani',
    'Ranjan', 'Rao', 'Rasheed', 'Rastogi', 'Rathod', 'Rathore', 'Rathour', 'Ratta', 'Rattan', 'Ratti',
    'Rau', 'Rauniyar', 'Rausa', 'Raut', 'Raval', 'Ravel', 'Ravi', 'Rawat', 'Rawlani', 'Ray',
    'Raykar', 'Raza', 'Reddy', 'Rege', 'Rehman', 'Rishiraj', 'Ritikesh', 'Rizvi', 'Rizwan', 'Roshan',
    'Rout', 'Roy', 'Rungta', 'Ruqaiya', 'Sabharwal', 'Sachan', 'Sachar', 'Sachdev', 'Sachdeva', 'Sagar',
    'Sah', 'Saha', 'Sahai', 'Sahni', 'Sahota', 'Sahu', 'Sain', 'Saini', 'Saluja', 'Salvi',
    'Sama', 'Samaddar', 'Samdharshni', 'Sami', 'Sampath', 'Samra', 'Sandal', 'Sandhu', 'Sandilya', 'Sane',
    'Sangha', 'Sanghvi', 'Sani', 'Sankar', 'Sankaran', 'Sankary', 'Sant', 'Sanwal', 'Saraf', 'Saran',
    'Saraswat', 'Sarath', 'Sarawagi', 'Sardar', 'Sareen', 'Sarin', 'Sarkar', 'Sarma', 'Sarna', 'Sarnobat',
    'Sarraf', 'Sasanka', 'Sasidharan', 'Sastry', 'Sathe', 'Savant', 'Sawhney', 'Sawlani', 'Saxena', 'Sehgal',
    'Sekh', 'Sekhar', 'Sekhon', 'Selvaraj', 'Sem', 'Sen', 'Sengar', 'Sengupta', 'Seshadri', 'Seth',
    'Sethi', 'Setty', 'Sha', 'Shah', 'Shahi', 'Shahid', 'Shaik', 'Shaikh', 'Shaji', 'Shan',
    'Shandilya', 'Shankar', 'Shankdhar', 'Shanker', 'Sharaf', 'Sharma', 'Shekhar', 'Shenoy', 'Shere', 'Sheth',
    'Shetty', 'Shinde', 'Shingh', 'Shiromani', 'Shivagauri', 'Shivastava', 'Shoaib', 'Shokeen', 'Shrivas', 'Shrivastav',
    'Shrivastava', 'Shroff', 'Shukla', 'Sibal', 'Siddiqui', 'Sidhu', 'Sikarwar', 'Singam', 'Singh', 'Singhal',
    'Singhania', 'Sinh', 'Sinha', 'Siraj', 'Sirohi', 'Sitaram', 'Sivan', 'Sodhi', 'Solanki', 'Som',
    'Soman', 'Sonawane', 'Sondhi', 'Soni', 'Sonkar', 'Sonker', 'Sonwani', 'Sood', 'Souza', 'Sridhar',
    'Srinivas', 'Srinivasan', 'Srivastav', 'Srivastava', 'Subbapati', 'Subramaniam', 'Subramanian', 'Suchi', 'Sugathan', 'Sule',
    'Suman', 'Sundar', 'Sundaram', 'Sunder', 'Sur', 'Sura', 'Surabhi', 'Suresh', 'Suri', 'Sutariya',
    'Swain', 'Swaminathan', 'Swamy', 'Tailor', 'Tak', 'Talwar', 'Tandon', 'Taneja', 'Tangirala', 'Tank',
    'Tanu', 'Tara', 'Tata', 'Tayde', 'Teja', 'Tejaswini', 'Tekchandani', 'Tella', 'Tewari', 'Tewary',
    'Thaker', 'Thakkar', 'Thakor', 'Thakur', 'Thaman', 'Thawani', 'Theratipally', 'Tibrewal', 'Tidke', 'Tiwari',
    'Tomar', 'Toor', 'Tripathi', 'Trivedi', 'Tulshidas', 'Tyagi', 'Umar', 'Unnikrishnan', 'Upadhyay', 'Uppal',
    'Upreti', 'Vaidya', 'Vaish', 'Vaishnavi', 'Vala', 'Varghese', 'Varkey', 'Varma', 'Varshney', 'Varty',
    'Varughese', 'Vasa', 'Vasav', 'Vashisth', 'Vasistha', 'Veerepalli', 'Venkataraman', 'Venkatesh', 'Venkateshwaran', 'Verma',
    'Vig', 'Vikram', 'Vilas', 'Virani', 'Virk', 'Vishal', 'Vishwakarma', 'Viswanathan', 'Vohra', 'Vora',
    'Vyas', 'Wable', 'Wadhawan', 'Wadhwa', 'Wagh', 'Wagle', 'Wali', 'Walia', 'Walla', 'Wanve',
    'Warrior', 'Warsi', 'Waseem', 'Wasif', 'Wason', 'Wathore', 'Waybhase', 'Waykos', 'Yadav', 'Yaddanapudi',
    'Yamala', 'Yogi', 'Yohannan', 'Zacharia', 'Zachariah', 'Zaidi', 'Zehra', 'Zope'
]


def get_initials(full_name):
    """Return list of first‑letter initials (lowercased) for each word in full_name."""
    return [word[0].lower() for word in full_name.strip().split()]

def can_match_all(initials1, initials2):
    """
    Return True if either list becomes empty after matching initials,
    regardless of order or leftovers in the other list.
    """
    i1 = [c.lower() for c in initials1]
    i2 = [c.lower() for c in initials2]

    # Match and remove shared initials
    for c in initials1:
        if c in i2:
            i2.remove(c)
            i1.remove(c)

    # Return true if either list is fully matched
    return len(i1) == 0 or len(i2) == 0

def names_match(name1, name2, last_name_list):
    # Split names into first and last using your logic
    first_name1, last_name1 = split_names(name1, last_name)
    first_name2, last_name2 = split_names(name2, last_name)

    # Reconstruct full names for comparison
    full_name1 = f"{first_name1} {last_name1}"
    full_name2 = f"{first_name2} {last_name2}"

    # Compare initials
    return can_match_all(get_initials(full_name1), get_initials(full_name2))




# Step 1: Remove titles
def remove_title(name):
    # Remove "Dr.", "dr", "DR", etc. from the beginning
    name = re.sub(r"^dr\.?\s*", "", name, flags=re.IGNORECASE)
    
    # Replace special characters with a space
    name = re.sub(r"[^\w\s]", " ", name)

    # Normalize multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    # Remove repeated words (case-insensitive)
    words = name.split()
    seen = set()
    result = []
    for word in words:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            result.append(word)
    
    name = " ".join(result)
    return name.strip()

# Step 2: Split names into first and last names
def split_names(name, last_names):
    name = remove_title(name)  # Remove any title from the name
    name_parts = name.split()  # Split the name into parts
    
    # Check if any last name from the list is present in the name
    last_name_found = None
    for part in name_parts:
        if part in last_names:
            last_name_found = part
            break
    
    # If a last name is found, split the name into first and last names
    if last_name_found:
        first_name = " ".join(name_parts[:-1])  # Everything except the last part is the first name
        last_name = last_name_found  # The last name
    else:
        first_name = name  # No split, the whole name is the first name
        last_name = ""  # No last name present
    
    return first_name.lower(), last_name.lower()

# Step 3: Compare first names
def check_first_name_similarity(first_name1, first_name2):
    try:
        # Encode the first names separately using the SentenceTransformer model
        embeddings1 = model.encode([first_name1])
        embeddings2 = model.encode([first_name2])

        # Calculate the cosine similarity between the two first names
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

        # Convert the similarity score to a Python native float
        similarity = float(similarity)

        # Set a similarity threshold (e.g., 0.8 for a high match)
        threshold = 0.7

        # If similarity is above the threshold, return 1 (indicating similar); otherwise, return 0
        result = 1 if similarity >= threshold else 0

        return {'result': result, 'similarity': similarity}

    except Exception as e:
        return {"error": str(e)}

# Step 4: Compare last names
def check_last_name_similarity(last_name1, last_name2):
    try:
        # Encode the last names separately using the SentenceTransformer model
        embeddings1 = model.encode([last_name1])
        embeddings2 = model.encode([last_name2])

        # Calculate the cosine similarity between the two last names
        similarity = cosine_similarity(embeddings1, embeddings2)[0][0]

        # Convert the similarity score to a Python native float
        similarity = float(similarity)

        # Set a similarity threshold (e.g., 0.8 for a high match)
        threshold = 0.7

        # If similarity is above the threshold, return 1 (indicating similar); otherwise, return 0
        result = 1 if similarity >= threshold else 0

        return {'result': result, 'similarity': similarity}

    except Exception as e:
        return {"error": str(e)}

# Step 5: Compare first and last names
def compare_names(name1, name2, last_name_list):
    # Check if the initials of both names match
    

    # Split the names into first and last names
    first_name1, last_name1 = split_names(name1, last_name_list)
    first_name2, last_name2 = split_names(name2, last_name_list)


    # Compare the first names
    first_name_similarity = check_first_name_similarity(first_name1, first_name2)

    # If both names have last names, compare them as well
    if last_name1 and last_name2:
        last_name_similarity = check_last_name_similarity(last_name1, last_name2)
        return first_name_similarity['result'] and last_name_similarity['result']
    else:
        # If one or both names don't have last names, only return the first name similarity
        return first_name_similarity['result']

    
    




# Load the Excel file

df = pd.read_excel(r"C:\Users\Desk0012\Downloads\Alekm_Final_Merging (2).xlsx")

# Step 6: Apply the comparison function to each row in the DataFrame
df['name_match'] = df.progress_apply(lambda row: compare_names(str(row['Doctor name']).lower(), str(row['client_name']).lower(), last_name), axis=1)
df['first_char_match'] = df.progress_apply(lambda row: names_match(str(row['Doctor name']).lower(), str(row['client_name']).lower(),last_name), axis=1)



#df['name_comparison_result for city'] = df.apply(lambda row: compare_names(str(row['city']).lower(), str(row['Client City']).lower(), last_name), axis=1)

# Save the updated DataFrame to a new Excel file
df.to_csv(r"C:\Users\Desk0012\Downloads\alkem_name_spec_150k_qc_unmatched_wala.csv", index=False)

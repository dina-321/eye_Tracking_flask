#model.py
from function import detect_cheating
import pickle

eye=detect_cheating()

# Save the model to a file
#with open('eye.pkl', 'wb') as f:
  #  pickle.dump(eye, f)
pickle.dump(eye, open('eye.pkl','wb'))

#with open('eye.pkl', 'rb') as f:
 #   eyee = pickle.load(f)
eyee = pickle.load(open('eye.pkl','rb'))

print(eyee)
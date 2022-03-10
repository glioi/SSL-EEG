import numpy as np
import torch
import torch.nn.functional as F


#################### Leave One Subject Out ############################
def loso(loso, number_of_subjects, path, device): 
    l= number_of_subjects 
    T_x=[]
    T_y=[]
    if loso==1 :
        loso_v=l
    else:
        loso_v=loso-1
    for subject_id in [e for e in range(1,l) if e not in (loso, loso_v)]:
        T_x.append(torch.load(path+str(subject_id)+'_x.pt').numpy())
        T_y.append(torch.load(path+str(subject_id)+'_y.pt').numpy())

    T_x=np.array(T_x, dtype=np.double).reshape([-1, 25, 1125])
    T_y=np.array(T_y, dtype=np.long).reshape([-1])

    T_x=torch.tensor(T_x).to(device)
    T_y=torch.tensor(T_y).to(device)

    V_x=torch.load(path+str(loso_v)+'_x.pt').to(device)
    V_y=torch.load(path+str(loso_v)+'_y.pt').to(device)

    Test_x=torch.load(path+str(loso)+'_x.pt').to(device)
    Test_y=torch.load(path+str(loso)+'_y.pt').to(device)


    
    return(T_x,T_y,V_x,V_y,Test_x,Test_y)

### train epoch
def train_epoch(network, optimizer, T_x, T_y, batch_size, tr_iter):
    cumu_loss = 0
    correct = 0.0
    total = 0.0
    data_perm = torch.randperm(T_x.shape[0])
    
    network.train()

    for i in range(T_x.shape[0] // batch_size + (1 if T_x.shape[0] % batch_size != 0 else 0)):
        data, target = T_x[data_perm[batch_size*i: batch_size*(i+1)]], T_y[data_perm[batch_size*i: batch_size*(i+1)]]        

        
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass 

        outputs = network(data)
       
        loss = F.cross_entropy(outputs, target)
        cumu_loss += loss.item()

      
    
        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()
        
        # compute accuracy
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += target.size(0)
        correct += (predicted == target).sum()


        tr_iter=tr_iter+1

    return cumu_loss / T_x.shape[0], correct/total, tr_iter

### validate epoch
def validate_epoch(network, T_x, T_y, batch_size, v_itr):
    cumu_loss = 0
    correct = 0.0
    total = 0.0
    data_perm = torch.randperm(T_x.shape[0])
    
    network.eval()
    
    with torch.no_grad():
    
        for i in range(T_x.shape[0] // batch_size + (1 if T_x.shape[0] % batch_size != 0 else 0)):
            data, target = T_x[data_perm[batch_size*i: batch_size*(i+1)]], T_y[data_perm[batch_size*i: batch_size*(i+1)]]   

            loss = F.cross_entropy(network(data), target)
            cumu_loss += loss.item()     

            # compute accuracy
            outputs = network(data)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)

            # Total number of labels
            total += target.size(0)
            correct += (predicted == target).sum()   

            v_itr=v_itr+1
    
    return cumu_loss / T_x.shape[0], correct/total, v_itr



### test epoch
def test(network, T_x, T_y, batch_size, n_classes, t_itr):
    # Calculate Accuracy
    correct = 0.0
    correct_arr = [0.0] * n_classes
    total = 0.0
    total_arr = [0.0] * n_classes
    y_true=[]
    y_pred=[]
    # Iterate through test dataset
    data_perm = torch.randperm(T_x.shape[0])
    
    network.eval()
    
    with torch.no_grad():
        for i in range(T_x.shape[0] // batch_size + (1 if T_x.shape[0] % batch_size != 0 else 0)):
            data, target = T_x[data_perm[batch_size*i: batch_size*(i+1)]], T_y[data_perm[batch_size*i: batch_size*(i+1)]]   

            outputs = network(data)
            
            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # Total number of labels
            total += target.size(0)
            correct += (predicted == target).sum()
            y_true.append(target)
            y_pred.append(predicted)
            
            for label in range(n_classes):
                correct_arr[label] += (((predicted == target) & (target==label)).sum())
                total_arr[label] += (target == label).sum()

    accuracy = correct / total
    print('TEST ACCURACY {} '.format(accuracy))
            
    t_itr=t_itr+1               
    return accuracy, t_itr

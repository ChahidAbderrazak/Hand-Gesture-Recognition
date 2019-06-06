#%% Classifers
#clf_model=SVC()
#tuned_parameters = [{'kernel': ['linear','rbf'], 'gamma': [1 ,2],'C': [ 1, 2]}]

#clf_model = LogisticRegression() #random_state=0, solver='lbfgs', multi_class='multinomial'
#tuned_parameters = [{'random_state': [0,1],'C':[1,5,10]}]#, 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]

clf_model =SGDClassifier()#(loss='hinge', alpha=0.0001, l1_ratio=0.15, epsilon=0.1, learning_rate='optimal', eta0=0.0, power_t=0.5, validation_fraction=0.1)

tuned_parameters = [{'loss': [ 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'] }]

import numpy as np


class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #(20,5)
        self._classes = np.unique(y)
        #y içindeki unique değerleri bulduk
        n_classes = len(self._classes)
        #init mean, var, priors

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        #bu ornekte iki farklı y oldupu için ve 5 tane toplam feature olduğu için  (2,5 matris)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        #y de bulunan 0 ve 1 in frekansı (y degeri(1.ci)/toplam sample)
        

        for c in self._classes:
            
            X_c = X[c==y]
            #X in içinde bulunan gözlemlerden hangileri 1 ve 0
            print("x_c", X_c)
            self._mean[c,:] = X_c.mean(axis=0)
            print("X_mean", self._mean)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] /float(n_samples)
            print("priors",self._priors, X_c.shape[0])

    def predict(self, X):

        y_pred = [self._predict(x) for x in X]
        #her bir x i al
        return y_pred


    def _predict(self ,x):

        posterios = []

        for idx, c in enumerate(self._classes):
            #her bi x değerini uniq y değeri kadar don
            print ("pred",idx, c)
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            print("class condi",class_conditional)
            posterior = prior + class_conditional
            posterios.append(posterior)
            #her x değerinin y değeri kadar hesapla ve en yüksek çıkan değeri bul
        print("pos",posterios)    
        print("aa",np.argmax(posterios))
        return self._classes[np.argmax(posterios)]      

    def _pdf(self, class_idx, x):
        print(x,class_idx)
        mean = self._mean[class_idx]
        print("mean",mean)
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2/(2*var))
        print("numerator", numerator)
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator






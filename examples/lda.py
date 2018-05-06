from np_ml import LDA, Documents

if __name__ == '__main__':
    print("data1")
    data = [["apple", "orange", "banana"], 
            ["apple", "orange"],
            ["orange", "banana"],
            ["cat", "dog"], 
            ["dog", "tiger"], 
            ["tiger", "cat"]]
    
    
    for i, document in enumerate(data):
        print("Document {}: {}".format(i, document))
    docs = Documents(data=data)
    lda = LDA()
    lda.fit(docs)
    print(docs.reverse_dict)
    print("theta (the probability of each topic for every document):")
    print(lda.theta)
    print("phi (the probability of each word for every topic):")
    print(lda.phi)
    print("")

    print("data2")
    data = [[1, 2, 3, 1, 2], 
            [1, 4, 5, 4, 4],
            [1, 4, 2, 5, 5, 4],
            [1, 3, 3, 2, 3],
            [1, 1, 3, 2, 2]]
    
    for i, document in enumerate(data):
        print("Document {}: {}".format(i, document))
    docs = Documents(data=data)
    lda.fit(docs)
    print(docs.reverse_dict)
    print("theta (the probability of each topic for every document):")
    print(lda.theta)
    print("phi (the probability of each word for every topic):")
    print(lda.phi)
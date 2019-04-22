"use strict";

const LOG_ON = true; //ERROR LOGGING STATUS 
const LOG_FREQ = 20000; //HOW OFTEN TO SHOW ERROR LOGS


class NeuralNetwork
{
    constructor(numInputs,numHidden,numOutputs)
    {
        this._numInputs=numInputs;
        this._numHidden=numHidden;
        this._numOutputs=numOutputs;
        this._hidden = [];
        this._inputs=[];

        this._bias0 = new Matrix(1,this._numHidden);
        this._bias1 = new Matrix(1,this._numOutputs);
        
        this._weights0 = new Matrix(this._numInputs,this._numHidden);
        this._weights1 = new Matrix(this._numHidden,this._numOutputs);

        //error logging
        this._logCount = LOG_FREQ;

        //randomize weights
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        this._weights0.randomWeights();
        this._weights1.randomWeights();
    }

    get bias0()
    {
        return this._bias0;
    }

    set bias0(bias0)
    {
        this._bias0 = bias0;
    }

    get bias1()
    {
        return this._bias1;
    }

    set bias1(bias1)
    {
        this._bias1 = bias1;
    }

    get inputs()
    {
        return this._inputs;
    }

    set inputs(inputs)
    {
        this._inputs = inputs;
    }

    get weights0()
    {
        return this._weights0;
    }

    set weights0(weights)
    {
        this._weights0 = weights;
    }

    get weights1()
    {
        return this._weights1;
    }

    set weights1(weights)
    {
        this._weights1 = weights;
    }

    get hidden()
    {
        return this._hidden;
    }

    set hidden(hidden)
    {
        this._hidden = hidden;
    }

    get logCount()
    {
        return this._logCount;
    }

    set logCount(logCount)
    {
        this._logCount = logCount;
    }
   


    feedForward(inputArray)
    {
        // convert input to matrix
        this.inputs = Matrix.convertFromArr(inputArray);
       

        // find hidden values and aply activation function
        this.hidden = Matrix.dot(this.inputs,this.weights0);
        this.hidden = Matrix.add(this.hidden,this.bias0);   //apply bias
        this.hidden = Matrix.map(this.hidden, x=> sigmoid(x));


        // find outputs values and aply activation function
        let outputs = Matrix.dot(this.hidden,this.weights1);
        outputs = Matrix.add(outputs,this.bias1);   //apply bias
        outputs = Matrix.map(outputs, x=> sigmoid(x));
        

        return outputs;

        
    }

    train(inputArray,targetArray)
    {
        //feed input data through the neural network
        let outputs = this.feedForward(inputArray);
        //console.log("outputs");
        //console.table(outputs.data);

        // calculate the output errors(target - output)
        let targets = Matrix.convertFromArr(targetArray);
        //console.log("targets");
        //console.table(targets.data);

        let outputErrors = Matrix.subtract(targets, outputs);
        //console.log("outputError");
        //console.table(outputErrors.data);

        //error logging
        if(LOG_ON)
        {
            if(this.logCount == LOG_FREQ)
            {
                console.log("error = " + outputErrors.data[0][0]);
            }
            this.logCount--;
            if(this.logCount==0)
            {
                this.logCount=LOG_FREQ;
            }
        }

        //calculate deltas(errors * derivatives of outputs)
        let outputDerivs = Matrix.map(outputs,x=> sigmoid(x,true));
        let outputDeltas = Matrix.multiply(outputErrors,outputDerivs);
        //console.log("outputDeltas");
        //console.table(outputDeltas.data);

        //calcuate hidden layer errors(deltas dot transpose of weights1)
        let weights1T= Matrix.transpose(this.weights1);
        let hiddenErrors = Matrix.dot(outputDeltas,weights1T);
        //console.log("hiddenErrors");
        //console.table(hiddenErrors.data);

        //calculate the hidden deltas (errors * derivative of hidden);
        let hiddenDerivs = Matrix.map(this.hidden,x=> sigmoid(x,true));
        let hiddenDeltas = Matrix.multiply(hiddenErrors,hiddenDerivs);
        //console.log("hiddenDeltas");
        //console.table(hiddenDeltas.data);

        //update the weights(add transpose of layers dot deltas)
        let hiddenT = Matrix.transpose(this.hidden);
        this.weights1 = Matrix.add(this.weights1,Matrix.dot(hiddenT,outputDeltas));
        let inputT = Matrix.transpose(this.inputs);
        this.weights0 = Matrix.add(this.weights0,Matrix.dot(inputT,hiddenDeltas));


        //update bias
        this.bias1 = Matrix.add(this.bias1,outputDeltas);
        this.bias0 = Matrix.add(this.bias0,hiddenDeltas);

    }
    
}


function sigmoid(x , deriv = false) {
    if(deriv)
    {
        return x*(1-x) ;        // where x = sigmoid(x)
    }
    return 1/(1+Math.exp(-x));        
}
/******************
 * Matrix Functions
 *******************/

class Matrix
{
    constructor(rows , cols , data = [])
    {
        this._rows=rows;
        this._cols=cols;
        this._data=data;

        //initialize with 0 if data not provided
        if(data == null|| data.length==0)
        {
            this._data = [];
            for(let i=0;i<this._rows;i++)
            {
                this._data[i] = [];
                for(let j=0;j<this._cols;j++)
                {
                    this._data[i][j]=0;
                }
            }
        }
        else    // check data integrity
        {
            if(data.length!=rows||data[0].length!=cols)
                throw new Error("Incorrect dimensions");
        }
    }

    get rows()
    {
        return this._rows
    }
    
    get cols()
    {
        return this._cols
    }
    
    get data()
    {
        return this._data
    } 
    
    // add two matrices
    static add( m0 , m1 )   // static to call it like a normal function operating on matrix types   
    {
         Matrix.checkDimesions(m0,m1);
         let m = new Matrix(m0.rows,m0.cols)

         for(let i=0;i<m.rows;i++)
         {
            for(let j=0;j<=m.cols;j++)
            {
                m.data[i][j]= m0.data[i][j]+m1.data[i][j];
            }
         }
         return m;
    }

    //check matrix have same dimensions 
    static checkDimesions(m0,m1)
    {
        if(m0.rows!=m1.rows || m0.cols!=m1.cols)
            throw new Error("matrix cannot be added");
    }


    static convertFromArr(arr)
    {
            return new Matrix(1,arr.length,[arr]);
    }
    // dot product of two matrices
    static dot(m0,m1)
    {
        if(m0.cols!=m1.rows)
        {
            //console.log(m0.cols + " " + m1.rows)
            throw new Error("we cannot dot product the matrices");
        }

        let m= new Matrix(m0.rows,m1.cols);
        
        for(let i=0;i<m.rows;i++)
        {
           for(let j=0;j<m.cols;j++)
           {
                let sum = 0;
                for(let k=0;k<m0.cols;k++)
                {
                    sum+=m0.data[i][k]*m1.data[k][j];
                }
                m.data[i][j]=sum;
           }
        }

        return m;
    }
    // apply a function to every cell of matrice
    static map(m0,mFunction)
    {
        let m = new Matrix(m0.rows,m0.cols);
        for(let i=0;i<m.rows;i++)
        {
           for(let j=0;j<m.cols;j++)
           {
                m.data[i][j]=mFunction(m0.data[i][j]);
                
           }
        }
        return m;
    }


    // find transpose of given matrix
    static transpose(m0) {
        let m = new Matrix(m0.cols, m0.rows);           //make no of rows = no of cols of transposed matrix
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m.data[j][i] = m0.data[i][j];
            }
        }
        return m;
    }

    //subtract two matrices
    static subtract( m0 , m1 )      
    {
         Matrix.checkDimesions(m0,m1);
         let m = new Matrix(m0.rows,m0.cols);

         for(let i=0;i<m.rows;i++)
         {
            for(let j=0;j<m.cols;j++)
            {
                m.data[i][j]= m0.data[i][j]-m1.data[i][j];
            }
         }
         return m;
    }

    // not the product of matrices instead just multiplying the respective elements of two matrices
    static multiply( m0 , m1 )    
    {
         Matrix.checkDimesions(m0,m1);
         let m = new Matrix(m0.rows,m0.cols)

         for(let i=0;i<m.rows;i++)
         {
            for(let j=0;j<m.cols;j++)
            {
                m.data[i][j]= m0.data[i][j]*m1.data[i][j];
            }
         }
         return m;
    }

    
    // apply random weights to the data table between -1 and 1
    randomWeights()
    {
        for(let i=0;i<this._rows;i++)
        {
            for(let j=0;j<this._cols;j++)
            {
                this.data[i][j]= Math.random() * 2 - 1;
            }
        }
    }
}
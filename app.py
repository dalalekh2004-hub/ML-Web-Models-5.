import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='ML Lab | Model Predictions', page_icon='🧠', layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.7rem;max-width:1200px}
.hero{padding:1.5rem 1.8rem;border-radius:18px;background:linear-gradient(120deg,#171d38,#323a68);color:white;margin-bottom:1rem}
.hero h1{color:white;margin:0}.hero p{opacity:.85;margin-bottom:0}
.result{padding:1rem;border-radius:14px;background:#f0f5ff;border:1px solid #d9e5ff}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def train_models():
    iris=load_iris(); diabetes=load_diabetes()
    # Binary Iris: Setosa vs Versicolor, petal length/width, notebook's exact selection
    xb=iris.data[iris.target != 2][:,2:4]; yb=iris.target[iris.target != 2]
    Xtr,Xte,ytr,yte=train_test_split(xb,yb,test_size=.2,random_state=42,stratify=yb)
    binary=Perceptron(max_iter=1000,eta0=.1,random_state=42).fit(Xtr,ytr)
    binary_acc=accuracy_score(yte,binary.predict(Xte))
    # Multi-class Iris OvR: standardized 4 features, 3 perceptrons
    Xtr_m,Xte_m,ytr_m,yte_m=train_test_split(iris.data,iris.target,test_size=.2,random_state=42,stratify=iris.target)
    scaler=StandardScaler().fit(Xtr_m); Xtr_s=scaler.transform(Xtr_m); Xte_s=scaler.transform(Xte_m)
    clfs=[]
    for k in range(3):
        clf=Perceptron(max_iter=1000,eta0=.1,random_state=42)
        clf.fit(Xtr_s,(ytr_m==k).astype(int)); clfs.append(clf)
    scores=np.column_stack([clf.decision_function(Xte_s) for clf in clfs])
    multi_acc=accuracy_score(yte_m,np.argmax(scores,axis=1))
    # Diabetes SLR using BMI only, MLR all ten dataset features
    # Fit actual notebook approaches
    Xd=diabetes.data[:,2].reshape(-1,1); yd=diabetes.target
    Xdtr,Xdte,ydtr,ydte=train_test_split(Xd,yd,test_size=.2,random_state=42)
    d_slr=LinearRegression().fit(Xdtr,ydtr)
    # Proper separate split and model for diabetes multiple
    Xdm_tr,Xdm_te,ydm_tr,ydm_te=train_test_split(diabetes.data,yd,test_size=.2,random_state=42)
    d_mlr=LinearRegression().fit(Xdm_tr,ydm_tr)
    # Iris regression target sepal length (first feature); SLR petal length (third), MLR features 2..4
    yi=iris.data[:,0]
    Xi_s=iris.data[:,2].reshape(-1,1)
    Xi_tr,Xi_te,yi_tr,yi_te=train_test_split(Xi_s,yi,test_size=.2,random_state=42)
    iris_slr=LinearRegression().fit(Xi_tr,yi_tr)
    Xi_m=iris.data[:,1:4]
    Xim_tr,Xim_te,yim_tr,yim_te=train_test_split(Xi_m,yi,test_size=.2,random_state=42)
    iris_mlr=LinearRegression().fit(Xim_tr,yim_tr)
    return dict(iris=iris,diabetes=diabetes,binary=binary,binary_acc=binary_acc,scaler=scaler,clfs=clfs,multi_acc=multi_acc,d_slr=d_slr,d_mlr=d_mlr,iris_slr=iris_slr,iris_mlr=iris_mlr)

M=train_models()

st.markdown('<div class="hero"><h1>Machine Learning Lab</h1><p>Train on the notebook datasets, enter new values, and get live predictions.</p></div>',unsafe_allow_html=True)
page=st.sidebar.radio('Choose a model section',['TP1 · Perceptron — Iris','TP2 · Linear Regression'])
st.sidebar.caption('Models are trained from the original scikit-learn datasets when the app starts. Predictions below use the trained models, not preset simulation values.')

# reusable chart

def metric_cards(vals):
    cols=st.columns(len(vals))
    for c,(label,value) in zip(cols,vals): c.metric(label,value)

if page.startswith('TP1'):
    st.header('🌸 Iris flower classification')
    st.write('The perceptron classifies Iris flowers from measurements. The flower photo shown below is a visual reference for the predicted class.')
    mode=st.radio('Classifier', ['Multiclass (3 species)','Binary (Setosa / Versicolor)'],horizontal=True)
    with st.form('iris_prediction'):
        st.subheader('Enter new flower measurements (cm)')
        if mode.startswith('Multi'):
            c1,c2=st.columns(2)
            with c1: sepal_l=st.number_input('Sepal length',min_value=0.0,max_value=10.0,value=5.1,step=.1)
            with c2: sepal_w=st.number_input('Sepal width',min_value=0.0,max_value=10.0,value=3.5,step=.1)
            c3,c4=st.columns(2)
            with c3: petal_l=st.number_input('Petal length',min_value=0.0,max_value=10.0,value=1.4,step=.1)
            with c4: petal_w=st.number_input('Petal width',min_value=0.0,max_value=10.0,value=.2,step=.1)
        else:
            petal_l=st.number_input('Petal length',min_value=0.0,max_value=10.0,value=1.4,step=.1)
            petal_w=st.number_input('Petal width',min_value=0.0,max_value=10.0,value=.2,step=.1)
        submitted=st.form_submit_button('🔍 Predict flower')
    if submitted:
        if mode.startswith('Multi'):
            arr=M['scaler'].transform([[sepal_l,sepal_w,petal_l,petal_w]])
            class_scores=np.array([clf.decision_function(arr)[0] for clf in M['clfs']])
            pred=int(np.argmax(class_scores)); name=M['iris'].target_names[pred]
            pics=['setosa.png','versicolor.png','virginica.png']
            st.markdown('<div class="result">',unsafe_allow_html=True)
            st.success(f'Prediction: **{name.title()}**')
            st.image('assets/'+pics[pred],caption=f'Predicted flower: {name.title()}',width=420)
            st.markdown('</div>',unsafe_allow_html=True)
            st.caption('One-vs-Rest decision scores (model output):')
            st.bar_chart(pd.DataFrame({'Score':class_scores},index=M['iris'].target_names))
            metric_cards([('Multiclass test accuracy',f'{M["multi_acc"]:.1%}')])
        else:
            pred=int(M['binary'].predict([[petal_l,petal_w]])[0]); name=M['iris'].target_names[pred]
            pic=['setosa.png','versicolor.png'][pred]
            st.success(f'Prediction: **{name.title()}**')
            st.image('assets/'+pic,caption=f'Predicted flower: {name.title()}',width=420)
            metric_cards([('Binary test accuracy',f'{M["binary_acc"]:.1%}')])
    st.info('Note: the photo is illustrative. The model prediction is based on the numerical measurements entered above.')

else:
    st.header('📈 Linear Regression')
    kind=st.selectbox('Choose regression model',['Diabetes · Simple (BMI only)','Diabetes · Multiple (10 features)','Iris · Simple (petal length → sepal length)','Iris · Multiple (3 features → sepal length)'])
    if kind.startswith('Diabetes · Simple'):
        st.markdown('Uses the Diabetes dataset and **BMI feature only**. BMI values in this dataset are standardized (small decimal values).')
        with st.form('diab_simple'):
            bmi=st.number_input('BMI feature (standardized)',min_value=-.2,max_value=.2,value=0.0,step=.01,format='%.4f')
            go=st.form_submit_button('Calculate prediction')
        if go:
            value=float(M['d_slr'].predict([[bmi]])[0]); st.metric('Predicted diabetes progression score',f'{value:.2f}')
            st.caption('This is a dataset target score—not a medical diagnosis.')
    elif kind.startswith('Diabetes · Multiple'):
        st.markdown('Enter the features for the sklearn Diabetes dataset. **Age** and **sex** accept familiar values and are converted to the dataset scale automatically; BMI, blood pressure, and serum features remain standardized dataset values.')
        feat=M['diabetes'].feature_names
        with st.form('diab_multiple'):
            inputs={}
            cols=st.columns(2)
            # Age and sex are accepted in familiar units and converted to the
            # normalized scale used by sklearn's diabetes dataset. The remaining
            # features are already normalized in this dataset and stay on that scale.
            age_years=None; sex_raw=None
            for i,f in enumerate(feat):
                with cols[i%2]:
                    if f=='age':
                        age_years=st.number_input('Age (years)',min_value=18,max_value=100,value=50,step=1)
                    elif f=='sex':
                        sex_raw=st.selectbox('Sex (dataset encoding)',options=[0,1],format_func=lambda x: '0' if x==0 else '1')
                    else:
                        inputs[f]=st.number_input(f'{f} (standardized)',min_value=-5.0,max_value=5.0,value=0.0,step=.01,format='%.4f')
            go=st.form_submit_button('Calculate prediction')
        if go:
            # sklearn Diabetes feature values are centered/scaled by std*sqrt(n).
            # These approximate original-unit conversions are only for age/sex.
            n_samples=len(M['diabetes'].data)
            age_scaled=(float(age_years)-48.6)/(13.1*np.sqrt(n_samples))
            sex_scaled=(float(sex_raw)-0.48)/(0.5*np.sqrt(n_samples))
            row=np.array([[age_scaled,sex_scaled]+[inputs[f] for f in feat if f not in ('age','sex')]])
            value=float(M['d_mlr'].predict(row)[0]); st.metric('Predicted diabetes progression score',f'{value:.2f}')
            st.caption('This is a dataset target score—not a medical diagnosis. BMI, BP, and s1–s6 must still be entered in standardized dataset scale.')
    elif kind.startswith('Iris · Simple'):
        st.markdown('Predict **sepal length (cm)** from petal length, using the Iris dataset.')
        with st.form('iris_slr'):
            petal=st.number_input('Petal length (cm)',min_value=0.0,max_value=10.0,value=3.0,step=.1)
            go=st.form_submit_button('Calculate prediction')
        if go:
            pred=float(M['iris_slr'].predict([[petal]])[0]); st.metric('Predicted sepal length (cm)',f'{pred:.2f}')
            fig,ax=plt.subplots(); ax.scatter([petal],[pred],s=100); ax.set_xlabel('Petal length (cm)'); ax.set_ylabel('Predicted sepal length (cm)'); ax.grid(True); st.pyplot(fig)
    else:
        st.markdown('Predict **sepal length (cm)** from sepal width, petal length, and petal width.')
        with st.form('iris_mlr'):
            a,b,c=st.columns(3)
            with a: sw=st.number_input('Sepal width (cm)',min_value=0.0,max_value=10.0,value=3.0,step=.1)
            with b: pl=st.number_input('Petal length (cm)',min_value=0.0,max_value=10.0,value=3.0,step=.1)
            with c: pw=st.number_input('Petal width (cm)',min_value=0.0,max_value=10.0,value=1.0,step=.1)
            go=st.form_submit_button('Calculate prediction')
        if go:
            pred=float(M['iris_mlr'].predict([[sw,pl,pw]])[0]); st.metric('Predicted sepal length (cm)',f'{pred:.2f}')
            fig,ax=plt.subplots(); ax.bar(['Sepal width','Petal length','Petal width'],[sw,pl,pw]); ax.set_ylabel('Entered measurements (cm)'); st.pyplot(fig)
    st.divider()
    st.subheader('Model quality (test set)')
    st.caption('For the selected dataset, the models use a fixed train/test split consistent with the notebook settings.')
    # generic quality estimates, fit actual held-out split for model choice
    if kind.startswith('Diabetes'):
        ds=load_diabetes(); target=ds.target
        if kind.startswith('Diabetes · Simple'):
            x=ds.data[:,2].reshape(-1,1)
        else: x=ds.data
        xtr,xte,ytr,yte=train_test_split(x,target,test_size=.2,random_state=42)
        model=LinearRegression().fit(xtr,ytr); pred=model.predict(xte)
    else:
        ir=load_iris(); target=ir.data[:,0]
        x=ir.data[:,2].reshape(-1,1) if kind.startswith('Iris · Simple') else ir.data[:,1:4]
        xtr,xte,ytr,yte=train_test_split(x,target,test_size=.2,random_state=42)
        model=LinearRegression().fit(xtr,ytr); pred=model.predict(xte)
    metric_cards([('MAE',f'{mean_absolute_error(yte,pred):.2f}'),('RMSE',f'{np.sqrt(mean_squared_error(yte,pred)):.2f}'),('R²',f'{r2_score(yte,pred):.3f}')])
    fig,ax=plt.subplots(); ax.scatter(yte,pred,alpha=.7); low=min(yte.min(),pred.min()); high=max(yte.max(),pred.max()); ax.plot([low,high],[low,high],linestyle='--'); ax.set_xlabel('Actual values'); ax.set_ylabel('Predicted values'); ax.set_title('Actual vs predicted'); ax.grid(True); st.pyplot(fig)

st.markdown('---')
st.caption('Academic demonstration based on the two uploaded notebooks. The Iris flower photos are illustrative; predictions come from trained sklearn models.')

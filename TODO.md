Testing
----------
- Split tests by module and reorder
- Add algebraic tests for test_calc_ewma
- Add aggregation tests for test_calc_ewma and test_calc_statistical_features


Performance
-----------
- Cache feature engineering results using lru.cache and profile to see if it makes a difference
- Clean-up warnings
- Update to_pandas, get_sample, and utilities functions to accept mxnet and Ray dataframes


Transform
-----------
- Split out two separate self.transforms to avoid mixing the sample and the data transforms. Self.transforms may not even be necessary anymore since cross_validate_lgbms stores the dicts locally
- Add a new "method" arg to encode_categoricals to allow the user to specifiy different strategies (e.g., get_dummies)


Feature engineering
-----------
- Write a function that flags consecutive days of non-sales
- Research enhancements in mlfin package from Advances in Financial Machine Learning
- Consider splitting out the crossover and momentum sections of calc_statistical_features and calc_ewma to reduce complexity. Current build may be faster than splitting the function separately due to matrix operations.
- calc_ewma, calc_percent.., and calc_statistical... could all be refactored; they share many moving parts.
- Ability to detrend features by different levels of the hierarchy (e.g., store/item sales divided by store slaes)
- Kurtosis and quantile features
- Innovation state space model features
- Shift as used in test__run_feature_engineering doesn't shift by day, but lag does correctly roll by day. this may cause minor issues if you don't fill your dataframe. 
- Add noise to features
  - # noise to time-static features
    # for col in [c for c in X.columns if 'store' in c and 'ratio' in c]:
    #     X[col] = X[col] + np.random.normal(0, 0.1, len(X))
    #     print('adding noise to {}'.format(col))
- Add new product forecasting features from https://hbswk.hbs.edu/item/how-do-you-predict-demand-and-set-prices-for-products-never-sold-before
- Add de-trended sales for each series, i.e. quantity sold divided by average quantity sold for that store, to capture item-level trends.
- Rolling averages by over time (by month, by week)
- Add quantiles
- Scale sales by dividing sales by store growth, including new rolling averages
- basic moving averages, after removing any store trends
    

Docs / README
-----------
- Add links to examples on README
- Add TOC to docs
- Set expectations for features like inventory, etc. (greatexpectations.io)
- Add package to pip
- Fix setup.py and distribute
- Need to add example notebook link to README

Modeling
-----------
- Add easy prediction capability with ensembling and add to docs
- Add recursiving training functionality (low priority since this concept can cause cascading errors)
- Build multi-quantile model using _get_quantile_weights
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/2.%20train/1-1.%20recursive_store_TRAIN.ipynb
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/2.%20train/1-2.%20recursive_store_cat_TRAIN.ipynb
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/3.%20predict/1-1.%20recursive_store_PREDICT.py
- Review additional M5 code for suggestions
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/2.%20train/2-2.%20nonrecursive_store_cat_TRAIN.ipynb
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/A1/3.%20code/3.%20predict/2-1.%20nonrecursive_store_PREDICT.py
    - Quantile modeling: https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/U1/quantiles_kaggle.ipynb
    - https://github.com/Mcompetitions/M5-methods/blob/master/Code%20of%20Winning%20Methods/U1/original/original_training.ipynb 
  - Add PurgedKFold classes    
      # class PurgedKFold(_BaseKFold):
      #     """
      #     Extend KFold class to work with labels that span intervals
      #     The train is purged of observations overlapping test-label intervals
      #     Test set is assumed contiguous (shuffle=False), w/o training samples in between
      #     """
      #     def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
      #         if not isinstance(t1,pd.Series):
      #             raise ValueError('Label Through Dates must be a pd.Series')
      #         super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
      #         self.t1=t1
      #         self.pctEmbargo=pctEmbargo
      #     def split(self,X,y=None,groups=None):
      #         if (X.index==self.t1.index).sum()!=len(self.t1):
      #             raise ValueError('X and ThruDateValues must have the same index')
      #         indices=np.arange(X.shape[0])
      #           mbrg=int(X.shape[0]*self.pctEmbargo)
      #         test_starts=[(i[0],i[-1]+1) for i in \
      #         np.array_split(np.arange(X.shape[0]),self.n_splits)]
      #         for i,j in test_starts:
      #             t0=self.t1.index[i] # start of test set
      #             test_indices=indices[i:j]
      #             maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
      #             train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
      #             if maxT1Idx<X.shape[0]: # right train (with embargo)
      #             train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:]))
      #             yield train_indices,test_indices


      # # grid search
      # def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.],
      #     n_jobs=-1,pctEmbargo=0,**fit_params):
      #     if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
      #     else:scoring='neg_log_loss' # symmetric towards all cases
      #     #1) hyperparameter search, on train data
      #     inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
      #     gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,
      #     scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
      #     gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
      #     #2) fit validated model on the entirety of the data
      #     if bagging[1]>0:
      #     gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps),
      #     n_estimators=int(bagging[0]),max_samples=float(bagging[1]),
      #     max_features=float(bagging[2]),n_jobs=n_jobs)
      #     gs=gs.fit(feat,lbl,sample_weight=fit_params \
      #     [gs.base_estimator.steps[-1][0]+'__sample_weight'])
      #     gs=Pipeline([('bag',gs)])
      #     return gs

      # # random search
      # def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.],
      # rndSearchIter=0,n_jobs=-1,pctEmbargo=0,**fit_params):
      # if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling
      # else:scoring='neg_log_loss' # symmetric towards all cases
      # #1) hyperparameter search, on train data
      # inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
      # if rndSearchIter==0:
      # gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,
      # scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
      # else:
      # 132 HYPER-PARAMETER TUNING WITH CROSS-VALIDATION
      # gs=RandomizedSearchCV(estimator=pipe_clf,param_distributions= \
      # param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,
      # iid=False,n_iter=rndSearchIter)
      # gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline
      # #2) fit validated model on the entirety of the data
      # if bagging[1]>0:
      # gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps),
      # n_estimators=int(bagging[0]),max_samples=float(bagging[1]),
      # max_features=float(bagging[2]),n_jobs=n_jobs)
      # gs=gs.fit(feat,lbl,sample_weight=fit_params \
      # [gs.base_estimator.steps[-1][0]+'__sample_weight'])
      # gs=Pipeline([('bag',gs)])
      # return gs


Interpretability
-----------
- Move error metric graphs from app to forecastframe
- Add feature to identify hierarchy levels (e.g., categories) that consistently over or under predict ("75% of cross-validation weeks were underpredicted for this category")
- Pass arg to make features orthogonal prior to feature importance
- Find a way around altair dataset size restrictions (see m5_example)
- The graphs in m5_example need work
- Add shap plots to package
- Needs better tests
- Being able to cluster products across levels of the hierarchy (given a keto trend in the West Coast, here are some other products in related categories that may be worth carrying)
library("recommenderlab")
library("Matrix")


NGOs <- read.csv("https://gist.githubusercontent.com/francescamanoni/ecab9b535f1af76360675d915975af65/raw/fcd4e6af3d364a000cf2d85945cfbe283713b7ad/NGO%2520names")

type <- sample(x = c("Education", "Health", "Elderly care","Children",
                     "Homeless", "Prisoners", "Animal care",
                     "Sports","People with Disabilities",
                     "Raising Awareness","Microfinance","Environmental"), size = 169, replace = T)


country <- read.csv("https://gist.githubusercontent.com/francescamanoni/cc1a6f234ac323db0de85e189b7d9164/raw/2d51597d461b738ea8e672f2ea2ce08b4b935666/Sub-Saharan%2520countries")


mydata <- matrix(sample(c(NA,0:5),169000, replace=TRUE, prob=c(.7,rep(.3/6,6))),
            nrow=1000, ncol= 169, dimnames = list(
              user=paste('user', 1:1000, sep=''),
              item= NGOs$Name
            ))

ngonames <- as.character (NGOs$Name)
countrynames <- as.character(country$Country)
typenames <- as.character(type)

mydf1 <- data.frame(  
  ngonames, countrynames, typenames) 



label <- c(paste("NGO", 1:131))
country <- c(paste("country",1:131))
type <- c(paste("type", 1:131))

mydf2 <- data.frame(
  label, country, type
)

colnames(mydf1) <- colnames(mydf2)
mydf <- rbind(mydf1, mydf2)

user_preference <- c("Education", "Health", "Elderly care","Children",
                     "Homeless", "Prisoners", "Animal care",
                     "Sports","People with Disabilities",
                     "Raising Awareness","Microfinance","Environmental")

mydf$type <- as.factor(mydf$type)


#creating a filter
filtered_recommend <- mydf[mydf$type %in% user_preference,1]
print(filtered_recommend) #for my users these are all the filtered NGOs

#use the rating matrix to find the recommendations

mydata <- as(mydata, "realRatingMatrix")

esSplit <- evaluationScheme(mydata, method="split", train=0.80,
                             given=-1, goodRating = 3)

getData(esSplit, "train")
getData(esSplit, "known")
getData(esSplit, "unknown")

#using UBCF algorithms 

UBCF_algorithms <- list(
  "UBCF_Jaccard" = list(name="ubcf", param = list(method = "Jaccard"))
)

#using Popularity algorithms

POP_algorithms <- list(
  "POPULARITY" = list(name="popular", param = list(method = "POPULAR"))
)


results_UBCF <- evaluate(esSplit, UBCF_algorithms, type = "topNList", n=c(1, 5, 10, 15))
plot(results_UBCF, annotate=c(1,2)) # we choose UBCF Jaccard


results_POP <- evaluate(esSplit, POP_algorithms, type = "topNList", n=c(1, 5, 10, 15))
plot(results_POP, annotate=c(1,2)) 

#recommendation for UBCF Model

recUBCF <- Recommender(getData(esSplit, "train"), method = "ubcf", param = list(method = "Jaccard", normalize = 'Z-Score'))

pUBCF_Jaccard <- predict(recUBCF, getData(esSplit, "known"), n = 5,  type = 'topNList')
UBCF <- calcPredictionAccuracy(pUBCF_Jaccard, getData(esSplit, "unknown"), given = 5, goodRating = 3)

print(UBCF)


#Popularity Models

recPop <- Recommender(getData(esSplit, "train"), method = "popular", param = list(method = "POPULAR", normalize = 'Z-Score'))

pPop <- predict(recPop, getData(esSplit, "known"), n = 5,  type = 'topNList')
Pop_1 <- calcPredictionAccuracy(pPop, getData(esSplit, "unknown"), given = 5, goodRating = 3)

print(Pop_1)


#PREDICTION OF THE FIRST TOP 5 NGO PROJECTS FOR ALL THE USERS BASED ON UBCF 
head(as(pUBCF_Jaccard, "list"))

#PREDICTION OF THE FIRST TOP  5 NGO PROJECTS FOR ALL THE USERS BASED ON POPULARITY
head(as(pPop, "list"))

#ENSEMBLE

Ensemble <- HybridRecommender(
  recPop,
  recUBCF,
  weights = c(0.4,0.6)
)

preEnsemble <- predict(Ensemble,getData(esSplit, "known"), n = 5,type = 'topNList')
preEnsemble

#PREDICTION OF THE FIRST TOP  5 NGO PROJECTS FOR ALL THE USERS BASED ON THE HYBRID RECOMMENDER
#(ENSEMBLE OF UBCF AND POPULARITY)
head(as(preEnsemble, "list"))






         
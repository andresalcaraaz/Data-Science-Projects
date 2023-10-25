library(tidyverse)
library(dplyr)
library(ggplot2)
library(caret)
library(MASS)
library(lmtest)
library(stats)
library(regclass)
library(car)
#library(shiny)
library(shinydashboard)
library(scales)
library(DT)

data = read.csv("/Users/andresalcaraz/Desktop/BDA 640 Final Project/Code & App/BDA640_FULLDATA")

#"ObservationRecordKey" - not really sure if we need to do anything here

#"Age" - good as is            

#"Gender"  
data$Gender <- as.factor(data$Gender)

#"PrimaryInsuranceCategory" 
data$PrimaryInsuranceCategory <- as.factor(data$PrimaryInsuranceCategory)

#"InitPatientClassAndFirstPostOUClass"
#data$InitPatientClassAndFirstPostOUClass <- as.factor(data$InitPatientClassAndFirstPostOUClass)

#"Flipped" - I'll create a factor variable to help out with visualization components   
data$flipped_factor <- as.factor(data$Flipped)

#"OU_LOS_hrs"  
data$OU_LOS_hrs <- as.numeric(data$OU_LOS_hrs)

#"DRG01" 
data$DRG01 <- as.factor(data$DRG01)
#renaming levels
levels(data$DRG01) <- c("Dehydration", "Congestive Heart Failure", "Pneumonia", "Colitis", "Pancreatitis", "GI Bleeding", "Urinary Tract Infection", "Syncope", "Edema", "Chest Pain", "Nausea", "Abdominal Pain")

#"BloodPressureUpper"     
data$BloodPressureUpper <- as.numeric(data$BloodPressureUpper)

#"BloodPressureLower" - Can probably just keep as is

#"BloodPressureDiff"
data$BloodPressureDiff <- as.numeric(data$BloodPressureDiff)

#"Pulse"             
data$Pulse <- as.numeric(data$Pulse)
summary(data$Pulse)

#"PulseOximetry"      
data$PulseOximetry <- as.numeric(data$PulseOximetry)

#"Respirations"       
data$Respirations <- as.numeric(data$Respirations)

#"Temperature"
data$Temperature <- as.numeric(data$Temperature)

#removing the not-so-useful column prior to model building
data$InitPatientClassAndFirstPostOUClass <- NULL
head(data)
colnames(data)
#Perform the split
set.seed(123)
sample_indices <- sample(1:nrow(data), size = floor(0.7*nrow(data)))
train <- data[sample_indices, ]
test <- data[-sample_indices, ]

data <- na.omit(data)

######## Final Logistic Model ##########
Final.Logit <- glm(Flipped ~ Age + Gender + PrimaryInsuranceCategory + DRG01 + Pulse, 
                   data = train, 
                   family = "binomial")
summary(Final.Logit)

######## Final Regression Model #########
#Full Model (log transform used to fix right skew in residuals)
LOS.Reg <- lm(log(OU_LOS_hrs) ~ Age + Gender + PrimaryInsuranceCategory + Flipped + DRG01 + BloodPressureDiff + 
                BloodPressureUpper + BloodPressureLower + Pulse + PulseOximetry + Respirations + Temperature, data = data)

#Step AIC
LOS.Step.Reg <- stepAIC(LOS.Reg, direction = "both")
summary(LOS.Step.Reg)

#FGLS
ehatsq <- resid(LOS.Step.Reg)^2
sighatsq.ols <- lm(log(ehatsq) ~ Age + PrimaryInsuranceCategory + 
                     Flipped + DRG01 + Pulse + Gender, data = data)
vari <- exp(fitted(sighatsq.ols))
LOS.FGLS <- lm(log(OU_LOS_hrs) ~ Age + PrimaryInsuranceCategory + Flipped + DRG01 + 
                 Pulse + Gender, weights = 1/vari, data = data)

# Initialize an empty vector to store the second predictions
second_prediction_values <- c()

# Initialize an action button
predict_button <- 0

# Define the UI

ui <- fluidPage(
  titlePanel("Predictive Model Dashboard"),
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "styles.css")
  ),
  sidebarLayout(
    sidebarPanel(
      h4("Input Parameters"),
      # Input elements for categorical variables
      selectInput("Gender", "Gender", choices = unique(data$Gender)),
      selectInput("DRG01", "DRG Code", choices = unique(data$DRG01)),
      selectInput("PrimaryInsuranceCategory", "Primary Insurance Category", choices = unique(data$PrimaryInsuranceCategory)),
      # Add more selectInput elements for other categorical variables
      sliderInput("Age", "Age", min = 0, max = 110, value = 30),
      sliderInput("Pulse", "Pulse Per Minute", min = 40, max = 220, value = 60),
      # Add more numericInput elements for other numerical variables
      actionButton("predict_button", "Predict", class = "action-button")
    ),
    mainPanel(
      h4("Prediction Results"),
      fluidRow(
        column(6, verbatimTextOutput("prediction_output")),
        column(6, verbatimTextOutput("second_prediction_output"))
      ),
      plotOutput("los_histogram"),
      h4("Summary Statistics"),
      dataTableOutput("summary_table")
    )
  )
)

# Function to calculate the mode
calculate_mode <- function(x) {
  uniq_x <- unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}

server <- function(input, output, session) {
  # Define user_data outside the reactive expressions
  user_data <- reactive({
    data.frame(
      Age = input$Age,
      Gender = input$Gender,
      PrimaryInsuranceCategory = input$PrimaryInsuranceCategory,
      DRG01 = input$DRG01,
      Pulse = input$Pulse
    )
  })
  
  # Reactive function to make predictions
  predicted_result <- reactive({
    req(input$predict_button)  # Ensure the button is clicked
    user_data_df <- user_data()  # Access user_data
    # Use the trained model to make predictions
    prediction1 <- predict(Final.Logit, newdata = user_data_df[, c("Age", "Gender", "PrimaryInsuranceCategory", "DRG01", "Pulse")], type = "response")
    return(prediction1)
  })
  
  # Reactive function to make the second prediction
  second_predicted_result <- reactive({
    req(input$predict_button)  # Ensure the button is clicked
    user_data_df <- user_data()  # Access user_data
    # Include the first prediction result in the user data
    user_data_df$Flipped <- predicted_result()
    
    # Use the second model to make predictions
    prediction2 <- predict(LOS.FGLS, newdata = user_data_df)
    
    # Store the second prediction in the vector
    second_prediction_values <<- c(second_prediction_values, prediction2)
    
    return(prediction2)
  })
  
  # Render the first prediction result
  output$prediction_output <- renderText({
    result <- predicted_result()
    paste("Probability of Flipping: ", scales::percent(result, accuracy = 0.01))
  })
  
  # Render the second prediction result
  output$second_prediction_output <- renderText({
    result2 <- second_predicted_result()
    if (length(result2) > 0) {
      paste("Length of Stay (Hours): ", scales::number(exp(result2), accuracy = 0.01))
    } else {
      "Length of Stay (Hours):"
    }
  })
  
  # Render the LOS histogram using ggplot2
  output$los_histogram <- renderPlot({
    if (length(second_prediction_values) > 0) {
      data <- data.frame(Length_of_Stay = exp(second_prediction_values))
      ggplot(data, aes(x = Length_of_Stay)) +
        geom_histogram(binwidth = 4, color = "black", fill = "#005BAD") +
        labs(title = "Length of Stay Histogram", x = "Length of Stay (Hours)", y = "Frequency") +
        theme_minimal() +
        theme(panel.background = element_rect(fill = "white"),
              plot.background = element_rect(fill = "white"))
    }
  })
  
  # Create a summary table for statistics
  output$summary_table <- renderDataTable({
    if (length(second_prediction_values) > 0) {
      stats <- data.frame(
        "Statistic" = c("Mean", "Mode", "Minimum", "First Quartile", "Median", "Third Quartile", "Maximum"),
        "Value" = c(
          scales::number(mean(exp(second_prediction_values)), accuracy = 0.01),
          scales::number(calculate_mode(exp(second_prediction_values)), accuracy = 0.01),
          scales::number(min(exp(second_prediction_values)), accuracy = 0.01),
          scales::number(quantile(exp(second_prediction_values), 0.25), accuracy = 0.01),
          scales::number(median(exp(second_prediction_values)), accuracy = 0.01),
          scales::number(quantile(exp(second_prediction_values), 0.75), accuracy = 0.01),
          scales::number(max(exp(second_prediction_values)), accuracy = 0.01)
        )
      )
      datatable(stats, options = list(dom = 't', paging = FALSE, searching = FALSE))
    }
  })
}


shinyApp(ui, server)

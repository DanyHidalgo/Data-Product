library(shiny)
library(ggplot2)
library(DBI)
library(RPostgres)
library(tidyr)
library(dplyr)
library(plotly)
library(jsonlite)


# Conectar 
con <- dbConnect(RPostgres::Postgres(),
                 dbname = "test",
                 host = "127.0.0.1",
                 port = 5432,
                 user = "test",
                 password = "test123")


# Server 

server <- function(input, output, session) {
  
  data <- reactive({
    query <- "
            SELECT DATE(timestamp) AS day, COUNT(timestamp) AS num_calls,
                   AVG(response_time) AS avg_response_time, COUNT(predictions) AS num_predictions
            FROM mage_ai.api_logs
            GROUP BY DATE(timestamp)
            ORDER BY day"
    dbGetQuery(con, query)
  })
  
  api_logs <- data.frame(
    request = c("/Data.json", "/Data.json", "/Data1.csv"),
    response = c('0', '{"predictions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}', '{"predictions": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'),
    timestamp = as.POSIXct(c("2024-11-15 18:06:36", "2024-11-15 18:37:05", "2024-11-15 21:27:17"))
  )
  
  output$plotCalls <- renderPlotly({
    p <- ggplot(data(), aes(x = day, y = num_calls)) +
      geom_bar(stat = "identity") +
      labs(title = "Llamadas por Día", x = "Día", y = "Número de Llamadas")
    ggplotly(p) %>%
      layout(dragmode = "select")
  })
  
  output$details <- renderPrint({
    eventdata <- event_data("plotly_click")
    if (!is.null(eventdata)) {
      day_selected <- data()$day[eventdata$pointNumber + 1]
      details <- data()[data()$day == day_selected,]
      paste("Día seleccionado: ", day_selected, "\n",
            "Tiempo promedio de respuesta: ", details$avg_response_time, "s\n",
            "Número de predicciones: ", details$num_predictions)
    } else {
      "Haga clic en una barra para ver detalles."
    }
  })

  
  ## Matriz de Confusión
  output$confusionMatrixPlot <- renderPlot({
    cm_data <- dbGetQuery(con, 'SELECT timestamp, confusion_matrix_ai, confusion_matrix_ad, confusion_matrix_bi, confusion_matrix_bd
                          FROM mage_ai.metrics
                          ORDER BY timestamp DESC
                          LIMIT 10')
    confusion_data_long <- pivot_longer(
      data = cm_data,
      cols = c(confusion_matrix_ai, confusion_matrix_ad, confusion_matrix_bi, confusion_matrix_bd),
      names_to = "Type",
      values_to = "Count"
    )
    
    confusion_data_long <- confusion_data_long %>%
      mutate(Type = recode(Type,
                           'confusion_matrix_ai' = 'True Positives',
                           'confusion_matrix_ad' = 'False Positives',
                           'confusion_matrix_bi' = 'False Negatives',
                           'confusion_matrix_bd' = 'True Negatives'))
    
    ggplot(data = confusion_data_long, aes(x = timestamp, y = Count, color = Type)) +
      geom_line() + 
      geom_point() + 
      labs(title = "Confusion Matrix Over Time ",
           x = "Timestamp",
           y = "Count",
           color = "Type") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
  })
  
  
  # Tabla de registros API
  output$api_logs_table <- renderDataTable({
    dbGetQuery(con, "SELECT * FROM mage_ai.api_logs ORDER BY Timestamp DESC")
  })
  
  
  # Gráficos de métricas del modelo
  output$metrics_plot <- renderPlot({
    metrics_data <- dbGetQuery(con, "SELECT timestamp, accuracy, precision, recall, f1_score, roc_auc FROM mage_ai.metrics ORDER BY Timestamp DESC")
    ggplot(metrics_data, aes(x = timestamp)) + 
      geom_line(aes(y = accuracy, color = "Accuracy")) +
      geom_line(aes(y = precision, color = "Precision")) +
      geom_line(aes(y = recall, color = "Recall")) +
      geom_line(aes(y = f1_score, color = "F1_Score")) +
      geom_line(aes(y = roc_auc, color = "Roc_Auc")) +
      labs(title = "Metrics Over Time", x = "Timestamp", y = "Score") +
      scale_color_manual(values = c("Accuracy" = "blue", "Precision" = "red", "Recall" = "green", "F1_Score" = "purple", "Roc_Auc"= "orange")) +
      theme_minimal()
  })
  
  
  ## Batch prediction
  output$table1 <- renderTable({
    inFile <- input$file1
    if (is.null(inFile)) {
      return(data.frame())
    }
    read.csv(inFile$datapath)
  })
  
  observeEvent(input$button, {
    data <- input$input1
    output$result <- renderText({
      paste("Resultado de predicción para:", data)
    })
  })  
  
  ## Atomic prediction
  output$result <- renderText({
    if(input$predict > 0) {
      paste("Resultado de la predicción:", input$time) 
    }
  })
  
  
  # Desconectar 
  session$onSessionEnded(function() {
    dbDisconnect(con)
  })
}



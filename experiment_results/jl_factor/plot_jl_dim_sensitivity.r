library(tidyverse)
library(patchwork)
library(ggtext)

pdf("jl_dim_sensitivity.pdf", width = 14, height = 10)

data <- read_csv("experiment_results.csv")

metadata <- read_csv("dataset_stats.csv") %>% 
  mutate(dataset = as.character(dataset))

make_meta_string <- function(meta_row) {
    ratio <- meta_row$num_edges / meta_row$num_nodes
  glue::glue(
    "Num nodes: {meta_row$num_nodes}\n",
    "Num edges: {meta_row$num_edges}\n",
    "Average degree: {sprintf('%.2f', ratio)}\n",
    "Notes: {meta_row$notes}"
  )
}

ggplot(data, aes(x = jl_factor, y = sparsification_rate * 100, color = sketch_type)) + 
geom_point() +
geom_line() +
scale_x_continuous(name ="JL factor", trans = "log2") +
scale_y_continuous(name = "Sparsification Rate (% edges remaining)") +
scale_color_discrete(name = "Sketch Type") +
ggtitle("Sparsification Rate as a function of JL factor") +
facet_wrap(~dataset)

ggplot(data, aes(x = jl_factor, color = sketch_type, fill = sketch_type)) + 
geom_bar(aes(y = (upper_bound_violations + lower_bound_violations)), stat = "identity", alpha = 0.5, position = "dodge") +
geom_line(aes(y = mean_rel_error*100)) + 
geom_point(aes(y = mean_rel_error*100)) +
scale_x_continuous(name ="JL factor", trans = "log2") +
scale_y_continuous(name = "Mean Relative Error (%)",
                  sec.axis = sec_axis(~. , name="Bound Violations (out of 100 trials)")) +
#scale_y_continuous(name = "Mean Relative Error") +

#scale_color_discrete(name = "Sketch Type") +
ggtitle("Quadratic Form experiment results - Real Mean Error and Bound Violations") +
facet_wrap(~dataset)

ggplot(data, aes(x = jl_factor, y = (evim_time + jl_time + solve_time + diff_norm_time + reweight_time), color = sketch_type)) +
geom_line() +
geom_point() +
scale_x_continuous(name ="JL factor", trans = "log2") +
scale_y_continuous(name = "Runtime (seconds)", trans = "log2"
#                  , limits = c(32, 256)
                  ) +
scale_color_discrete(name = "Sketch Type") +
ggtitle("Runtime as a function of JL factor") +
facet_wrap(~dataset)

data %>% 
  group_split(dataset) %>%                     # list of data‑frames, one per dataset
  walk(function(d_sub) {
    ds <- unique(d_sub$dataset)
    meta_row <- metadata %>% filter(dataset == ds) %>% slice(1)

    # ---- Sparsification Rate ---------------------------------
    p_rate <- ggplot(d_sub, aes(x = jl_factor, y = sparsification_rate * 100, color = sketch_type)) + 
        geom_point(size = 2) +
        geom_line(linewidth = 1) +
        scale_x_continuous(name ="JL factor", trans = "log2") +
        scale_y_continuous(name = "Sparsification Rate (% edges remaining)") +
        scale_color_discrete(name = "Sketch Type")

    # ---- Error rate -----------------------------
    p_error1 <- ggplot(d_sub, aes(x = jl_factor, color = sketch_type, fill = sketch_type)) + 
        geom_line(aes(y = mean_rel_error), linetype = "dashed", linewidth = 1) + 
        geom_point(aes(y = mean_rel_error), size = 2) +
        scale_x_continuous(name ="JL factor", trans = "log2") +
        scale_y_continuous(name = "Mean Relative Error") +
        scale_color_discrete(name = "Sketch Type")
    
    # ----- Out of bounds events ---------
    p_error2 <- ggplot(d_sub, aes(x = jl_factor, color = sketch_type, fill = sketch_type)) + 
        geom_bar(aes(y = (upper_bound_violations + lower_bound_violations)), stat = "identity", alpha = 0.5, position = "dodge") +
        scale_x_continuous(name ="JL factor", trans = "log2") +
        scale_y_continuous(name = "Bound Violations (out of 100 trials)",
                            breaks=seq(0, 100, 25),
                            limits=c(0,100))

    # ---- Runtime --------------------------------------------------------------
    p_time <- ggplot(d_sub, aes(x = jl_factor, y = (evim_time + jl_time + solve_time + diff_norm_time + reweight_time), color = sketch_type)) +
        geom_line(linetype = "dotdash", linewidth = 1) +
        geom_point(size = 2) +
        scale_x_continuous(name ="JL factor", trans = "log2") +
        scale_y_continuous(name = "Runtime (seconds)") +
        scale_color_discrete(name = "Sketch Type")
    
    # ---- Dataset details ----------------------------------------------------
    meta_txt <- make_meta_string(meta_row)

    meta_plot <- ggplot() +
    geom_textbox(
        aes(x = 0.5, y = 0.5, label = meta_txt),
        width = unit(0.95, "npc"),        # max width = 95 % of the panel
        halign = 0, valign = 1,
        size = 4, lineheight = 1.2,
        fill = "gray95", colour = "gray70",
        box.colour = NA,
        box.padding = margin(0,0,0,0, "pt")
    ) +
    scale_x_continuous(expand = c(0,0), limits = c(0,1)) +
    scale_y_continuous(expand = c(0,0), limits = c(0,1)) +
    theme_void()

    # ---- Stack the four plots ----------------------
    combined <- (meta_plot | p_time | p_rate) /
               (p_error1 | p_error2)   + 
               plot_layout(guides = "collect") +
               plot_annotation(
                    title = ds,
                    theme = theme(
                        plot.title = element_text(
                            size = 16, face = "bold", hjust = 0.5
                        ), 
                        legend.position = "bottom",      # put the shared legend at bottom
                        legend.title    = element_blank()
                    )
                )

    # Send the combined plot to the PDF device
    print(combined)
  })

  dev.off()
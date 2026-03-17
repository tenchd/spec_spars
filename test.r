library(tidyverse)

data <- read_csv("basic_experiment_results_with_fscore.csv")

ggplot(data, aes(x = epsilon, y = sparsification_rate * 100, color = sketch_type)) + 
geom_point() +
geom_line() +
scale_x_continuous(name ="Epsilon", 
                    breaks=seq(0,1,0.25)) +
scale_y_continuous(name = "Sparsification Rate (% edges remaining)") +
scale_color_discrete(name = "Sketch Type") +
facet_wrap(~dataset)

ggplot(data, aes(x = epsilon, color = sketch_type, fill = sketch_type)) + 
geom_bar(aes(y = (upper_bound_violations + lower_bound_violations)), stat = "identity", alpha = 0.5, position = "dodge") +
geom_line(aes(y = mean_rel_error*100)) + 
geom_point(aes(y = mean_rel_error*100)) +
scale_x_continuous(name ="Epsilon", 
                    breaks=seq(0,1,0.25)) +
scale_y_continuous(name = "Mean Relative Error (%)",
                  sec.axis = sec_axis(~. , name="Bound Violations (out of 100 trials)")) +
#scale_y_continuous(name = "Mean Relative Error") +

#scale_color_discrete(name = "Sketch Type") +
facet_wrap(~dataset)

ggplot(data, aes(x = epsilon, color = sketch_type, fill = sketch_type)) + 
geom_line(aes(y = fscore), linetype = "dotted") +
geom_point(aes(y = fscore)) +
geom_line(aes(y = mean_rel_error)) + 
geom_point(aes(y = mean_rel_error)) +
scale_x_continuous(name ="Epsilon", 
                    breaks=seq(0,1,0.25)) +
scale_y_continuous(name = "Mean Relative Error",
                  sec.axis = sec_axis(~. , name="fscore")) +
#scale_y_continuous(name = "Mean Relative Error") +

#scale_color_discrete(name = "Sketch Type") +
facet_wrap(~dataset)

ggplot(data, aes(x = epsilon, y = (evim_time + jl_time + solve_time + diff_norm_time + reweight_time), color = sketch_type)) +
geom_line() +
geom_point() +
scale_x_continuous(name ="Epsilon", 
                    breaks=seq(0,1,0.25)) +
scale_y_continuous(name = "Runtime (seconds)", trans = "log2"
#                  , limits = c(32, 256)
                  ) +
scale_color_discrete(name = "Sketch Type") +
facet_wrap(~dataset)
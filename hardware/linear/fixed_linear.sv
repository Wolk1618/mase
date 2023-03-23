module fixed_linear #(
    parameter IN_WIDTH = 32,
    parameter IN_SIZE  = 4,
    parameter IN_DEPTH = 3,

    parameter WEIGHT_WIDTH = 16,
    parameter WEIGHT_SIZE  = IN_SIZE,

    parameter B_WIDTH = IN_WIDTH + WEIGHT_WIDTH + $clog2(IN_SIZE) + $clog2(IN_DEPTH),
    parameter PARALLELISM = 2,

    // This is the width for the summed product
    // +1 is because of the bias
    parameter HAS_BIAS  = 0,
    parameter OUT_WIDTH = B_WIDTH + HAS_BIAS,
    parameter OUT_SIZE  = PARALLELISM

) (
    input clk,
    input rst,

    // input port for data_inivations
    input  [IN_WIDTH-1:0] data_in      [IN_SIZE-1:0],
    input                 data_in_valid,
    output                data_in_ready,

    // input port for weight
    input  [WEIGHT_WIDTH-1:0] weight      [IN_SIZE*PARALLELISM-1:0],
    input                     weight_valid,
    output                    weight_ready,

    /* verilator lint_off UNUSEDSIGNAL */
    input  [  B_WIDTH-1:0] bias          [OUT_SIZE-1:0],
    input                  bias_valid,
    /* verilator lint_on UNUSEDSIGNAL */
    output                 bias_ready,
    output [OUT_WIDTH-1:0] data_out      [OUT_SIZE-1:0],
    output                 data_out_valid,
    input                  data_out_ready
);

  localparam FDP_WIDTH = IN_WIDTH + WEIGHT_WIDTH + $clog2(IN_SIZE);
  localparam ACC_WIDTH = FDP_WIDTH + $clog2(IN_DEPTH);

  logic fdp_join_valid, fdp_join_ready;
  join2 #() fdp_join_inst (
      .data_in_ready ({weight_ready, data_in_ready}),
      .data_in_valid ({weight_valid, data_in_valid}),
      .data_out_valid(fdp_join_valid),
      .data_out_ready(fdp_join_ready)
  );

  /* verilator lint_off UNUSEDSIGNAL */
  // Assume the parallelised hardware above have the same arrival time
  // which means that they always have the same state. So we can just
  // pick one of the valid signal to use.
  logic [PARALLELISM-1:0] fdp_data_ready, fdp_weight_ready;
  assign fdp_join_ready = fdp_data_ready[0];
  /* verilator lint_on UNUSEDSIGNAL */

  logic                 acc_ready;
  logic [ACC_WIDTH-1:0] acc_data_out[PARALLELISM-1:0];

  // There are PARALLELISM number of dot product instances with IN_SIZE inputs 
  // and each one computes for IN_DEPTH iterations for each inputs.
  for (genvar i = 0; i < PARALLELISM; i = i + 1) begin : linear
    // Assume the weight are transposed and partitioned 
    logic [WEIGHT_WIDTH-1:0] current_weight[WEIGHT_SIZE-1:0];
    assign current_weight = weight[WEIGHT_SIZE*i+WEIGHT_SIZE-1:WEIGHT_SIZE*i];

    logic [FDP_WIDTH-1:0] fdp_data_out;
    logic fdp_data_out_valid, fdp_data_out_ready;

    // The inputs are already sync-ed by the previous join
    fixed_dot_product #(
        .IN_WIDTH(IN_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .IN_SIZE(IN_SIZE)
    ) fdp_inst (
        .clk(clk),
        .rst(rst),
        .data_in(data_in),
        .data_in_valid(fdp_join_valid),
        .data_in_ready(fdp_data_ready[i]),
        .weight(current_weight),
        .weight_valid(fdp_join_valid),
        .weight_ready(fdp_weight_ready[i]),
        .data_out(fdp_data_out),
        .data_out_valid(fdp_data_out_valid),
        .data_out_ready(fdp_data_out_ready)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    logic acc_data_out_valid, acc_data_out_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    fixed_accumulator #(
        .IN_WIDTH(FDP_WIDTH),
        .IN_DEPTH(IN_DEPTH)
    ) fixed_accumulator_inst (
        .clk(clk),
        .rst(rst),
        .data_in(fdp_data_out),
        .data_in_valid(fdp_data_out_valid),
        .data_in_ready(fdp_data_out_ready),
        .data_out(acc_data_out[i]),
        .data_out_valid(acc_data_out_valid),
        .data_out_ready(acc_data_out_ready)
    );

    // Assume the parallelised hardware above have the same arrival time
    // which means that they always have the same state. So we can just
    // pick one of the valid signal to use.
    assign acc_data_out_ready = acc_ready;
  end


  if (HAS_BIAS == 1) begin

    logic acc_join_valid, acc_join_ready;
    join2 #() acc_join_inst (
        .data_in_ready ({bias_ready, acc_ready}),
        .data_in_valid ({bias_valid, linear[0].acc_data_out_valid}),
        .data_out_valid(acc_join_valid),
        .data_out_ready(acc_join_ready)
    );
    logic [PARALLELISM-1:0] reg_ready;
    assign acc_join_ready = &reg_ready;

    for (genvar i = 0; i < PARALLELISM; i = i + 1) begin : add_bias
      logic [OUT_WIDTH-1:0] add;
      logic [ACC_WIDTH-1:0] bias_sext;
      // Sign extend
      assign bias_sext = {{(ACC_WIDTH - B_WIDTH) {bias[i][B_WIDTH-1]}}, bias[i]};
      assign add = acc_data_out[i] + bias_sext;
      logic dout_valid;
      register_slice #(
          .IN_WIDTH(OUT_WIDTH),
      ) register_slice (
          .clk           (clk),
          .rst           (rst),
          .data_in_valid (acc_join_valid),
          .data_in_ready (reg_ready[i]),
          .data_in_data  (add),
          .data_out_valid(dout_valid),
          .data_out_ready(data_out_ready),
          .data_out_data (data_out[i])
      );
    end
    assign data_out_valid = add_bias[0].dout_valid;

  end else begin
    assign acc_ready = data_out_ready;
    assign data_out_valid = linear[0].acc_data_out_valid;
    assign data_out = acc_data_out;
    assign bias_ready = 1;
  end

endmodule
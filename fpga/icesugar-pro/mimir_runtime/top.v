module top (
    input wire clk25,
    input wire serial_rx,
    output wire serial_tx
);
    localparam integer CLOCK_HZ = 25_000_000;
    localparam integer BAUD = 115_200;
    localparam integer CLOCKS_PER_BIT = CLOCK_HZ / BAUD;
    localparam integer FIFO_DEPTH = 32;

    reg [15:0] rx_clock = 0;
    reg [3:0] rx_bit = 0;
    reg [7:0] rx_shift = 0;
    reg rx_busy = 0;
    reg rx_ready = 0;
    reg rx_meta = 1;
    reg rx_sync = 1;

    reg [15:0] tx_clock = 0;
    reg [3:0] tx_bit = 0;
    reg [9:0] tx_shift = 10'b11_1111_1111;
    reg tx_busy = 0;
    reg tx_line = 1;

    reg [7:0] tx_fifo [0:FIFO_DEPTH - 1];
    reg [4:0] fifo_write = 0;
    reg [4:0] fifo_read = 0;
    reg [5:0] fifo_count = 0;

    reg [3:0] caps_match = 0;
    reg [2:0] dot_match = 0;
    reg [2:0] load_match = 0;
    reg [2:0] exec_match = 0;
    reg [2:0] matrix_load_match = 0;
    reg [2:0] matrix_exec_match = 0;
    reg [1:0] dot_state = 0;
    reg [6:0] dot_length = 0;
    reg [6:0] dot_index = 0;
    reg signed [7:0] dot_lhs = 0;
    reg signed [31:0] dot_accumulator = 0;
    reg signed [31:0] dot_result = 0;

    reg signed [7:0] resident_weights [0:63];
    reg [1:0] load_state = 0;
    reg [6:0] load_length = 0;
    reg [6:0] load_index = 0;
    reg [6:0] resident_length = 0;
    reg [1:0] exec_state = 0;
    reg [6:0] exec_length = 0;
    reg [6:0] exec_index = 0;
    reg signed [31:0] exec_accumulator = 0;

    reg signed [7:0] matrix_weights [0:511];
    reg signed [31:0] matrix_accumulators [0:7];
    reg signed [31:0] matrix_results [0:7];
    reg [1:0] matrix_load_state = 0;
    reg [3:0] matrix_rows = 0;
    reg [6:0] matrix_columns = 0;
    reg [9:0] matrix_load_index = 0;
    reg [1:0] matrix_exec_state = 0;
    reg [6:0] matrix_exec_index = 0;
    integer matrix_row;

    reg response_active = 0;
    reg [2:0] response_kind = 0;
    reg [5:0] response_index = 0;
    reg [7:0] response_byte = 0;

    wire fifo_pop = !tx_busy && fifo_count != 0;
    wire fifo_push = response_active && (fifo_count != FIFO_DEPTH || fifo_pop);

    assign serial_tx = tx_line;

    always @(*) begin
        response_byte = 0;
        if (response_kind == 0) begin
            case (response_index)
                0: response_byte = "M";
                1: response_byte = "I";
                2: response_byte = "M";
                3: response_byte = "I";
                4: response_byte = "R";
                5: response_byte = "F";
                6: response_byte = "P";
                7: response_byte = "G";
                8: response_byte = 8'd1;
                9: response_byte = 8'd0;
                10: response_byte = 8'd7;
                11, 12, 13: response_byte = 8'd0;
                14: response_byte = 8'd64;
                default: response_byte = 8'd0;
            endcase
        end else if (response_kind == 1) begin
            case (response_index)
                0: response_byte = "D";
                1: response_byte = "R";
                2: response_byte = "E";
                3: response_byte = "S";
                4: response_byte = dot_result[7:0];
                5: response_byte = dot_result[15:8];
                6: response_byte = dot_result[23:16];
                default: response_byte = dot_result[31:24];
            endcase
        end else if (response_kind == 2) begin
            case (response_index)
                0: response_byte = "W";
                1: response_byte = "A";
                2: response_byte = "C";
                default: response_byte = "K";
            endcase
        end else if (response_kind == 3) begin
            if (response_index == 0) response_byte = "M";
            else if (response_index == 1) response_byte = "R";
            else if (response_index == 2) response_byte = "E";
            else if (response_index == 3) response_byte = "S";
            else if (response_index == 4) response_byte = matrix_rows;
            else begin
                case ((response_index - 5) & 3)
                    0: response_byte = matrix_results[(response_index - 5) >> 2][7:0];
                    1: response_byte = matrix_results[(response_index - 5) >> 2][15:8];
                    2: response_byte = matrix_results[(response_index - 5) >> 2][23:16];
                    default: response_byte = matrix_results[(response_index - 5) >> 2][31:24];
                endcase
            end
        end else begin
            case (response_index)
                0: response_byte = "M";
                1: response_byte = "A";
                2: response_byte = "C";
                default: response_byte = "K";
            endcase
        end
    end

    always @(posedge clk25) begin
        rx_meta <= serial_rx;
        rx_sync <= rx_meta;
        rx_ready <= 0;

        if (!rx_busy) begin
            if (!rx_sync) begin
                rx_busy <= 1;
                rx_clock <= CLOCKS_PER_BIT / 2;
                rx_bit <= 0;
            end
        end else if (rx_clock == 0) begin
            rx_clock <= CLOCKS_PER_BIT - 1;
            if (rx_bit == 0) begin
                if (rx_sync) rx_busy <= 0;
                else rx_bit <= 1;
            end else if (rx_bit <= 8) begin
                rx_shift[rx_bit - 1] <= rx_sync;
                rx_bit <= rx_bit + 1;
            end else begin
                rx_busy <= 0;
                if (rx_sync) rx_ready <= 1;
            end
        end else begin
            rx_clock <= rx_clock - 1'b1;
        end

        if (rx_ready && dot_state == 0 && load_state == 0 && exec_state == 0 &&
            matrix_load_state == 0 && matrix_exec_state == 0 && !response_active) begin
            case (caps_match)
                0: caps_match <= (rx_shift == "M") ? 1 : 0;
                1: caps_match <= (rx_shift == "I") ? 2 : 0;
                2: caps_match <= (rx_shift == "M") ? 3 : 0;
                3: caps_match <= (rx_shift == "I") ? 4 : 0;
                4: caps_match <= (rx_shift == "R") ? 5 : 0;
                5: caps_match <= (rx_shift == "?") ? 6 : 0;
                6: caps_match <= (rx_shift == 8'd1) ? 7 : 0;
                7: begin
                    caps_match <= 0;
                    if (rx_shift == 8'h0a) begin
                        response_active <= 1;
                        response_kind <= 0;
                        response_index <= 0;
                    end
                end
            endcase

            case (dot_match)
                0: dot_match <= (rx_shift == "D") ? 1 : 0;
                1: dot_match <= (rx_shift == "O") ? 2 : 0;
                2: dot_match <= (rx_shift == "T") ? 3 : 0;
                3: begin
                    dot_match <= 0;
                    if (rx_shift == "8") dot_state <= 1;
                end
                default: dot_match <= 0;
            endcase
            case (load_match)
                0: load_match <= (rx_shift == "L") ? 1 : 0;
                1: load_match <= (rx_shift == "O") ? 2 : 0;
                2: load_match <= (rx_shift == "A") ? 3 : 0;
                3: begin
                    load_match <= 0;
                    if (rx_shift == "D") begin
                        load_state <= 1;
                        dot_match <= 0;
                        exec_match <= 0;
                        caps_match <= 0;
                    end
                end
                default: load_match <= 0;
            endcase
            case (exec_match)
                0: exec_match <= (rx_shift == "E") ? 1 : 0;
                1: exec_match <= (rx_shift == "X") ? 2 : 0;
                2: exec_match <= (rx_shift == "E") ? 3 : 0;
                3: begin
                    exec_match <= 0;
                    if (rx_shift == "C") exec_state <= 1;
                end
                default: exec_match <= 0;
            endcase
            case (matrix_load_match)
                0: matrix_load_match <= (rx_shift == "M") ? 1 : 0;
                1: matrix_load_match <= (rx_shift == "W") ? 2 : 0;
                2: matrix_load_match <= (rx_shift == "G") ? 3 : 0;
                3: begin
                    matrix_load_match <= 0;
                    if (rx_shift == "T") begin
                        matrix_load_state <= 1;
                        caps_match <= 0;
                        dot_match <= 0;
                        load_match <= 0;
                        exec_match <= 0;
                        matrix_exec_match <= 0;
                    end
                end
                default: matrix_load_match <= 0;
            endcase
            case (matrix_exec_match)
                0: matrix_exec_match <= (rx_shift == "M") ? 1 : 0;
                1: matrix_exec_match <= (rx_shift == "V") ? 2 : 0;
                2: matrix_exec_match <= (rx_shift == "E") ? 3 : 0;
                3: begin
                    matrix_exec_match <= 0;
                    if (rx_shift == "C") begin
                        matrix_exec_state <= 1;
                        caps_match <= 0;
                        dot_match <= 0;
                        load_match <= 0;
                        exec_match <= 0;
                        matrix_load_match <= 0;
                    end
                end
                default: matrix_exec_match <= 0;
            endcase
        end else if (rx_ready && dot_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64) begin
                dot_length <= rx_shift[6:0];
                dot_index <= 0;
                dot_accumulator <= 0;
                dot_state <= 2;
            end else begin
                dot_state <= 0;
            end
        end else if (rx_ready && dot_state == 2) begin
            dot_lhs <= rx_shift;
            dot_state <= 3;
        end else if (rx_ready && dot_state == 3) begin
            if (dot_index + 1'b1 == dot_length) begin
                dot_result <= dot_accumulator + dot_lhs * $signed(rx_shift);
                dot_state <= 0;
                response_active <= 1;
                response_kind <= 1;
                response_index <= 0;
            end else begin
                dot_accumulator <= dot_accumulator + dot_lhs * $signed(rx_shift);
                dot_index <= dot_index + 1'b1;
                dot_state <= 2;
            end
        end else if (rx_ready && load_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64) begin
                load_length <= rx_shift[6:0];
                load_index <= 0;
                load_state <= 2;
            end else begin
                load_state <= 0;
            end
        end else if (rx_ready && load_state == 2) begin
            resident_weights[load_index] <= rx_shift;
            if (load_index + 1'b1 == load_length) begin
                resident_length <= load_length;
                load_state <= 0;
                response_active <= 1;
                response_kind <= 2;
                response_index <= 0;
            end else begin
                load_index <= load_index + 1'b1;
            end
        end else if (rx_ready && exec_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 64 && rx_shift[6:0] == resident_length) begin
                exec_length <= rx_shift[6:0];
                exec_index <= 0;
                exec_accumulator <= 0;
                exec_state <= 2;
            end else begin
                exec_state <= 0;
            end
        end else if (rx_ready && exec_state == 2) begin
            if (exec_index + 1'b1 == exec_length) begin
                dot_result <= exec_accumulator +
                    resident_weights[exec_index] * $signed(rx_shift);
                exec_state <= 0;
                response_active <= 1;
                response_kind <= 1;
                response_index <= 0;
            end else begin
                exec_accumulator <= exec_accumulator +
                    resident_weights[exec_index] * $signed(rx_shift);
                exec_index <= exec_index + 1'b1;
            end
        end else if (rx_ready && matrix_load_state == 1) begin
            if (rx_shift != 0 && rx_shift <= 8) begin
                matrix_rows <= rx_shift[3:0];
                matrix_load_state <= 2;
            end else begin
                matrix_load_state <= 0;
            end
        end else if (rx_ready && matrix_load_state == 2) begin
            if (rx_shift == 64) begin
                matrix_columns <= rx_shift[6:0];
                matrix_load_index <= 0;
                matrix_load_state <= 3;
            end else begin
                matrix_load_state <= 0;
            end
        end else if (rx_ready && matrix_load_state == 3) begin
            matrix_weights[matrix_load_index] <= rx_shift;
            if (matrix_load_index + 1'b1 == matrix_rows * matrix_columns) begin
                matrix_load_state <= 0;
                response_active <= 1;
                response_kind <= 4;
                response_index <= 0;
            end else begin
                matrix_load_index <= matrix_load_index + 1'b1;
            end
        end else if (rx_ready && matrix_exec_state == 1) begin
            if (rx_shift == matrix_columns && matrix_rows != 0) begin
                matrix_exec_index <= 0;
                for (matrix_row = 0; matrix_row < 8; matrix_row = matrix_row + 1)
                    matrix_accumulators[matrix_row] <= 0;
                matrix_exec_state <= 2;
            end else begin
                matrix_exec_state <= 0;
            end
        end else if (rx_ready && matrix_exec_state == 2) begin
            for (matrix_row = 0; matrix_row < 8; matrix_row = matrix_row + 1) begin
                if (matrix_row < matrix_rows) begin
                    if (matrix_exec_index + 1'b1 == matrix_columns)
                        matrix_results[matrix_row] <= matrix_accumulators[matrix_row] +
                            matrix_weights[matrix_row * 64 + matrix_exec_index] * $signed(rx_shift);
                    else
                        matrix_accumulators[matrix_row] <= matrix_accumulators[matrix_row] +
                            matrix_weights[matrix_row * 64 + matrix_exec_index] * $signed(rx_shift);
                end
            end
            if (matrix_exec_index + 1'b1 == matrix_columns) begin
                matrix_exec_state <= 0;
                response_active <= 1;
                response_kind <= 3;
                response_index <= 0;
            end else begin
                matrix_exec_index <= matrix_exec_index + 1'b1;
            end
        end

        if (!tx_busy) begin
            tx_line <= 1;
            if (fifo_count != 0) begin
                tx_shift <= {1'b1, tx_fifo[fifo_read], 1'b0};
                tx_bit <= 0;
                tx_clock <= CLOCKS_PER_BIT - 1;
                tx_line <= 0;
                tx_busy <= 1;
            end
        end else if (tx_clock == 0) begin
            if (tx_bit == 9) begin
                tx_busy <= 0;
                tx_line <= 1;
            end else begin
                tx_bit <= tx_bit + 1'b1;
                tx_shift <= {1'b1, tx_shift[9:1]};
                tx_line <= tx_shift[1];
                tx_clock <= CLOCKS_PER_BIT - 1;
            end
        end else begin
            tx_clock <= tx_clock - 1'b1;
        end

        if (fifo_push) begin
            tx_fifo[fifo_write] <= response_byte;
            fifo_write <= fifo_write + 1'b1;
            if ((response_kind == 0 && response_index == 15) ||
                (response_kind == 1 && response_index == 7) ||
                (response_kind == 2 && response_index == 3) ||
                (response_kind == 3 && response_index + 1'b1 == 5 + matrix_rows * 4) ||
                (response_kind == 4 && response_index == 3)) begin
                response_active <= 0;
            end else begin
                response_index <= response_index + 1'b1;
            end
        end
        if (fifo_pop) fifo_read <= fifo_read + 1'b1;
        case ({fifo_push, fifo_pop})
            2'b10: fifo_count <= fifo_count + 1'b1;
            2'b01: fifo_count <= fifo_count - 1'b1;
            default: fifo_count <= fifo_count;
        endcase
    end
endmodule
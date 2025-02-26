module example (input a, b, c, output y);
    wire w1, w2;

    assign w1 = a & b;
    assign w2 = w1 | c;
    assign y = ~w2;
endmodule

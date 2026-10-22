% PROBLEMA 1
syms t1 t2 t3 t4 t5 t6 real

DH = @(t,r,a,d) [cos(t), -sin(t)*cos(r),  sin(t)*sin(r), a*cos(t);
    sin(t),  cos(t)*cos(r), -cos(t)*sin(r), a*sin(t);
    0,          sin(r),          cos(r),        d;
    0,                0,                0,        1];


A1 = simplify(DH(t1,sym(pi)/2,0,50));

A2 = simplify(DH(t2 + sym(pi)/2,0,93,0));

A3 = simplify(DH(t3,0,93,0));

A4 = simplify(DH(t4,0,50,0));

T03 = simplify(A1*A2*A3);

T04 = simplify(T03*A4);
pretty(T04)

T04_0 = double(subs(T04, [t1 t2 t3 t4], [0 0 0 0]))

H45 = [1 0 0 -15;
    0 1 0 45;
    0 0 1 0;
    0 0 0 1];

T05 = T04 * H45;


% PROBLEMA 2
syms x4 y4 z4 x4z real

theta1 = atan2(y4, x4);
q1 = theta1

h = sqrt(1 - x4z^2);            

o3 = [x4 - 50 * cos(q1)*h; 
    y4 - 50 * sin(q1)*h; 
    z4 - 50 * x4z];

xc = o3(1); 
yc = o3(2); 
zc = o3(3);


r = sqrt(xc^2 + yc^2)
s = zc - 50

cos3 = (r^2 + s^2 - 93^2 - 93^2) / (2*93*93)
sin3 = -sqrt(1 - cos3^2);

theta3 = atan2(sin3, cos3);
q3 = theta3

theta2 = atan2(s, r) - atan2(93*sin3, 93 + 93*cos3);
q2 = theta2 - pi/2

%Partint del que hem trobat abans de que x4z = cos(q2 + q3 + q4)
q4 = -acos(x4z) - q2 - q3

q = [q1; q2; q3; q4];

% Convertim les expressions simboliques en funcions numeriques
inversa = matlabFunction(q,   'Vars', {x4, y4, z4, x4z});
directa = matlabFunction(T04, 'Vars', {[t1; t2; t3; t4]});

% PROBLEMA 3

R  = 32;
pc = [150; 0; 120];
N  = 36;
Q  = zeros(N+1, 4);

for j = 0:N
    phi = j * 2*pi/N;
    p = pc + R * [0; cos(phi); sin(phi)];
    
    qj = inversa(p(1), p(2), p(3), 0);
    Q(j+1, 1) = qj(1);
    Q(j+1, 2) = qj(2);
    Q(j+1, 3) = qj(3);
    Q(j+1, 4) = qj(4);

    % Comprovacio amb la directa
    Tj = directa(qj);
    fprintf('j=%2d  err = %.1e  r31 = %.1e\n', j, norm(Tj(1:3,4) - p), Tj(3,1));
end

% PROBLEMA 4

T01 = A1;
T02 = simplify(A1 * A2);

o0 = [0; 0; 0];
o1 = T01(1:3,4);
o2 = T02(1:3,4);
o3 = T03(1:3,4);
o4 = T04(1:3,4);
o5 = T05(1:3,4);

z0 = [0; 0; 1];
z1 = T01(1:3,3);
z2 = T02(1:3,3);
z3 = T03(1:3,3);

J4 = [cross(z0, o4-o0) cross(z1, o4-o1) cross(z2, o4-o2) cross(z3, o4-o3);
    z0, z1, z2, z3];

J5 = [cross(z0, o5-o0) cross(z1, o5-o1) cross(z2, o5-o2) cross(z3, o5-o3);
    z0, z1, z2, z3];

J4 = simplify(J4);
J5 = simplify(J5);


J4_num = matlabFunction(J4, 'Vars',{[t1; t2; t3; t4]}); 
J5_num = matlabFunction(J5, 'Vars',{[t1; t2; t3; t4]});

for j = [0 9 18 27]
    qj = Q(j+1,:).';
    J4_def = J4_num(qj);
    J5_def = J5_num(qj);
end

% PROBLEMA 5

syms vx vy vz vf real

x4_esp = T04(1:3,1);

Jv = J4(1:3,:);
Jw = J4(4:6,:);

A = [Jv;
    x4_esp(2)*Jw(1,:) - x4_esp(1)*Jw(2,:)];

v = [vx; vy; vz; vf];

q_dot = A \ v;

vel_joints = matlabFunction(q_dot, 'Vars', {[t1; t2; t3; t4], [vx; vy; vz; vf]});

q_dot5 = vel_joints(Q(10,:).', [0; -3; 0; 0])

% PROBLEMA 6

t0 = 0;
tf = 2;
M = [t0^5     t0^4     t0^3    t0^2  t0  1;
     5*t0^4   4*t0^3   3*t0^2  2*t0  1   0;
     20*t0^3  12*t0^2  6*t0    2     0   0;
     tf^5     tf^4     tf^3    tf^2  tf  1;
     5*tf^4   4*tf^3   3*tf^2  2*tf  1   0;
     20*tf^3  12*tf^2  6*tf    2     0   0];

knots = [0 9 18 27 36];
V = [0   0   0;
     0 -27   0;
     0   0 -27;
     0  27   0;
     0   0   0];
coef = zeros(4,6,4);

for s = 1:4
    q_ini    = Q(knots(s)+1,:).';
    qd_ini   = vel_joints(q_ini, [V(s,:).'; 0]);
    qdd_ini  = zeros(4,1);

    q_fi     = Q(knots(s+1)+1,:).';
    qd_fi    = vel_joints(q_fi, [V(s+1,:).'; 0]);
    qdd_fi   = zeros(4,1);

    for i = 1:4
        b = [q_ini(i); qd_ini(i); qdd_ini(i); q_fi(i); qd_fi(i); qdd_fi(i)];
        coef(i,:,s) = (inv(M) * b).';
    end
end

A6 = coef(:,:,1);
B6 = coef(:,:,2);
C6 = coef(:,:,3);
D6 = coef(:,:,4);

% comprovation of correcntess. q1 of b has to be the same as q1 of a when
% instant of a = 2 and instant of b = 0
for i = 1:4
    fprintf('joint %d:  A(t=2) = %.6f   B(t=0) = %.6f\n', i, polyval(A6(i,:), 2), polyval(B6(i,:), 0));
end

% PROBLEMA 7

coefs = {A6, B6, C6, D6};
t = linspace(0, 8, 801);
P = zeros(3, numel(t));
for k = 1:numel(t)
    s  = min(floor(t(k)/2) + 1, 4);
    tl = t(k) - 2*(s-1);
    Cs = coefs{s};
    qt = zeros(4,1);
    for i = 1:4
        qt(i) = polyval(Cs(i,:), tl);
    end
    Tt = directa(qt);
    P(:,k) = Tt(1:3,4);
end

phi = linspace(0, 2*pi, 361);
Pd  = pc + R * [zeros(1,361); cos(phi); sin(phi)];

N  = 12;
Ts = 8 / N;
w  = 2*pi / 8;
t0 = 0;
tf = Ts;
M = [t0^5     t0^4     t0^3    t0^2  t0  1;
     5*t0^4   4*t0^3   3*t0^2  2*t0  1   0;
     20*t0^3  12*t0^2  6*t0    2     0   0;
     tf^5     tf^4     tf^3    tf^2  tf  1;
     5*tf^4   4*tf^3   3*tf^2  2*tf  1   0;
     20*tf^3  12*tf^2  6*tf    2     0   0];

knots = 0 : 36/N : 36;
Vk = zeros(N+1, 3);
for k = 2:N
    phi = knots(k) * 2*pi/36;
    Vk(k,:) = R * w * [0, -sin(phi), cos(phi)];
end

coefN = zeros(4,6,N);
for s = 1:N
    q_ini   = Q(knots(s)+1,:).';
    qd_ini  = vel_joints(q_ini, [Vk(s,:).'; 0]);
    qdd_ini = zeros(4,1);
    q_fi    = Q(knots(s+1)+1,:).';
    qd_fi   = vel_joints(q_fi, [Vk(s+1,:).'; 0]);
    qdd_fi  = zeros(4,1);
    for i = 1:4
        b = [q_ini(i); qd_ini(i); qdd_ini(i); q_fi(i); qd_fi(i); qdd_fi(i)];
        coefN(i,:,s) = (inv(M) * b).';
    end
end

t  = linspace(0, 8, 801);
PN = zeros(3, numel(t));
for k = 1:numel(t)
    s  = min(floor(t(k)/Ts) + 1, N);
    tl = t(k) - Ts*(s-1);
    qt = zeros(4,1);
    for i = 1:4
        qt(i) = polyval(coefN(i,:,s), tl);
    end
    Tt = directa(qt);
    PN(:,k) = Tt(1:3,4);
end

figure
plot(Pd(2,:), Pd(3,:), 'b--'); hold on
plot(P(2,:),  P(3,:),  'r')
plot(PN(2,:), PN(3,:), 'g')
axis equal; grid on
xlabel('y_0 [mm]'); ylabel('z_0 [mm]')
legend('Cercle desitjat', '5 knot-points', sprintf('%d knot-points', N+1))


% PROBLEMA 8
% Al llarg del cami del problema 3
c_cami = zeros(1, 37);
for j = 1:37
    c_cami(j) = cond(J4_num(Q(j,:).'));
end
phi_j = (0:36) * 2*pi/36;

% Al llarg de la trajectoria del problema 6
coefs  = {A6, B6, C6, D6};
t      = linspace(0, 8, 801);
c_traj = zeros(1, numel(t));
for k = 1:numel(t)
    s  = min(floor(t(k)/2) + 1, 4);
    tl = t(k) - 2*(s-1);
    Cs = coefs{s};
    qt = zeros(4,1);
    for i = 1:4
        qt(i) = polyval(Cs(i,:), tl);
    end
    c_traj(k) = cond(J4_num(qt));
end

figure
subplot(2,1,1)
plot(phi_j, c_cami, 'o-'); grid on
xlabel('\phi [rad]'); ylabel('cond(J_4)')
title('Cami del problema 3')
subplot(2,1,2)
plot(t, c_traj); grid on
xlabel('t [s]'); ylabel('cond(J_4)')
title('Trajectoria del problema 6')

% PROBLEMA 9

F = [0; 0; -1; 0; 0; 0];

torques = zeros(4, 37);

for j = 1:37
    J4j = J4_num(Q(j,:).');
    torques(:, j) = J4j.' * F;
end

figure
for i = 1:4
    subplot(4,1,i)
    plot(phi_j, torques(i,:), 'o-'); grid on
    ylabel(sprintf('\\tau_%d', i))
end
xlabel('\phi [rad]')

% PROBLEMA 10

%definim primer els centres de masses
d1 = [0;  -20; 0];
d2 = [-30;  0; 0];
d3 = [-30;  0; 0];
d4 = [-25; 15; 0];

rc1 = T01(1:3,4) + T01(1:3,1:3) * d1;
rc2 = T02(1:3,4) + T02(1:3,1:3) * d2;
rc3 = T03(1:3,4) + T03(1:3,1:3) * d3;
rc4 = T04(1:3,4) + T04(1:3,1:3) * d4;

%ara toca definir totes les Jacobianes dels links

Z = zeros(3,1);

Jvc1 = [cross(z0,rc1-o0), Z,                Z,                Z               ];
Jvc2 = [cross(z0,rc2-o0), cross(z1,rc2-o1), Z,                Z               ];
Jvc3 = [cross(z0,rc3-o0), cross(z1,rc3-o1), cross(z2,rc3-o2), Z               ];
Jvc4 = [cross(z0,rc4-o0), cross(z1,rc4-o1), cross(z2,rc4-o2), cross(z3,rc4-o3)];

Jvc1 = simplify(Jvc1/1000);
Jvc2 = simplify(Jvc2/1000);
Jvc3 = simplify(Jvc3/1000);
Jvc4 = simplify(Jvc4/1000);

Jw1 = [z0, Z,  Z,  Z ];
Jw2 = [z0, z1, Z,  Z ];
Jw3 = [z0, z1, z2, Z ];
Jw4 = [z0, z1, z2, z3];

%definim els tensors d'inercia. Tenim els locals que ens dona el enunciat i
%calculem I0 a partir dalla

syms I0 real

m1 = 0.06;  
m2 = 0.08;  
m3 = 0.08;  
m4 = 0.04;

I1_loc = I0 * diag([1    0.4  0.9]);
I2_loc = I0 * diag([0.45 1.4  1.2]);
I3_loc = I0 * diag([0.45 1.4  1.2]);
I4_loc = I0 * diag([0.5  0.5  0.5]);

syms a2 c2 real
h = 0.050;

eq1 = m1/12 * (h^2 + c2) == I0;
eq2 = m1/12 * (a2  + c2) == 0.4*I0;
eq3 = m1/12 * (a2  + h^2) == 0.9*I0;

sol = solve([eq1, eq2, eq3], [a2, c2, I0]);

I0_val = double(sol.I0);

%convertim tots aquests tensors locals a els tensors globals amb el valor
%de I0 que hem trobat abans
I1_loc = double(subs(I1_loc, I0, I0_val));
I2_loc = double(subs(I2_loc, I0, I0_val));
I3_loc = double(subs(I3_loc, I0, I0_val));
I4_loc = double(subs(I4_loc, I0, I0_val));

R1 = T01(1:3,1:3);
R2 = T02(1:3,1:3);
R3 = T03(1:3,1:3);
R4 = T04(1:3,1:3);

I1 = simplify(R1 * I1_loc * R1.');
I2 = simplify(R2 * I2_loc * R2.');
I3 = simplify(R3 * I3_loc * R3.');
I4 = simplify(R4 * I4_loc * R4.');

%definim les matrius d'inercia de cada joint

D1 = m1 * (Jvc1.' * Jvc1) + Jw1.' * I1 * Jw1;
D2 = m2 * (Jvc2.' * Jvc2) + Jw2.' * I2 * Jw2;
D3 = m3 * (Jvc3.' * Jvc3) + Jw3.' * I3 * Jw3;
D4 = m4 * (Jvc4.' * Jvc4) + Jw4.' * I4 * Jw4;

D = simplify(D1 + D2 + D3 + D4);

%calculem els simbols de cristoffel

syms dq1 dq2 dq3 dq4 real

qs  = [t1; t2; t3; t4];
dqs = [dq1; dq2; dq3; dq4];

n  = 4;
cs = sym(zeros(n, n, n));
C  = sym(zeros(n, n));

for k = 1:n
    for j = 1:n
        for i = 1:n
            cs(i,j,k) = 1/2 * ( diff(D(k,j), qs(i)) ...
                + diff(D(k,i), qs(j)) ...
                - diff(D(i,j), qs(k)) );
          
            C(k,j) = C(k,j) + cs(i,j,k) * dqs(i);
        end
    end
end

%i ara calculem els vetors de gravetat

gvec = [0; 0; -9.81];


Ep = -( m1 * gvec.' * rc1 ...
    + m2 * gvec.' * rc2 ...
    + m3 * gvec.' * rc3 ...
    + m4 * gvec.' * rc4 ) / 1000;

G = sym(zeros(4,1));
for k = 1:4
    G(k) = diff(Ep, qs(k));
end
G = simplify(G); 

%finalment muntem la formula final en format matricial

syms ddq1 ddq2 ddq3 ddq4 real
ddqs = [ddq1; ddq2; ddq3; ddq4];

tau_inercia  = D * ddqs;
tau_coriolis = C * dqs;
tau_gravetat = G;

tau_sym = tau_inercia + tau_coriolis + tau_gravetat;

tau_num = matlabFunction(tau_sym, 'Vars', {qs, dqs, ddqs});

%fem el plot

coefs = {A6, B6, C6, D6};
t     = linspace(0, 8, 801);
tau   = zeros(4, numel(t));

for k = 1:numel(t)
    s  = min(floor(t(k)/2) + 1, 4);
    tl = t(k) - 2*(s-1);
    Cs = coefs{s};

    qt   = zeros(4,1);
    dqt  = zeros(4,1);
    ddqt = zeros(4,1);
    for i = 1:4
        qt(i)   = polyval(Cs(i,:), tl);
        dqt(i)  = polyval(polyder(Cs(i,:)), tl);
        ddqt(i) = polyval(polyder(polyder(Cs(i,:))), tl);
    end

    tau(:,k) = tau_num(qt, dqt, ddqt);
end

% Plot
figure
for i = 1:4
    subplot(4,1,i)
    plot(t, tau(i,:), 'b'); grid on
    ylabel(sprintf('\\tau_%d [N·m]', i))
    xline([2 4 6], ':k');                      % canvis de segment
end
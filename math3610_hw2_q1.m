% MATH 3610 HW 2 Problem 1

% Description:
% This program uses linear programming to find
% optimal/minimum study hours given a semester & major gpa goal.
% 
% View more detailed write-up in the pdf submitted.

%%%%% some constants %%%%%

% course credits for math3610, math2210, bionb3300, stsci2200, respectively
cred = [4 4 4 4];

% semester & major gpa goal
sem_goal = 3.0;
maj_goal = 3.5; % NOTE: assume that i am a math major

% available hours for studying every week
idle_hr = 50.8;

% grade if did no study for each course with full background: 
% aced all quizzes & exams,
% but did not do any assignments & projects
gr_0 = [0.75 0.9 0.4 0.81];

% interests, ability, background, course expectation
math3610 = [8 3 2 8];
math2210 = [7 3 3 7];
bionb3300 = [8 5 6 6];
stsci2200 = [4 7 3 3];

assert(all(1 <= math3610 & math3610 <= 10),"need to be in range of 1 and 10, inclusive")
assert(all(1 <= math2210 & math2210 <= 10),"need to be in range of 1 and 10, inclusive")
assert(all(1 <= bionb3300 & bionb3300 <= 10),"need to be in range of 1 and 10, inclusive")
assert(all(1 <= stsci2200 & stsci2200 <= 10),"need to be in range of 1 and 10, inclusive")

courses = [math3610; math2210; bionb3300; stsci2200];

%%%%% some helper functions to calculate course grade & upper bound %%%%%

function [const,slope]=hr_to_gr(course_id,courses,gr_0,cred)
    %{
    create a function for course grade as a function of study hour per week

    course_id (int): indicates which course
    courses (list): contains a list of courses i am taking
    gr_0 (list): max course grade if aced all exams, but no homework submitted
    cred (list): number of credits for each course
    %}

    % my interests, ability, background, course expectation to the course
    course_stat = courses(course_id,:);
    
    % calculate beta (rescale background)
    beta_i = 10/gr_0(course_id);

    % grade as a function of study hour per week (x)
    % index 1 is constant, index 2 is slope
    const = 4.3*cred(course_id)*course_stat(3)/beta_i;
    slope = 4.3*cred(course_id)*(1/34)*((course_stat(1)+course_stat(2))/course_stat(4));
end

function upper_bound=calc_ub(course_id,courses,gr_0)
    %{
    calculates upper bound, when Gr(x) = 1

    course_id (int): indicates which course
    courses (list): contains a list of courses i am taking
    gr_0 (list): max course grade if aced all exams, but no homework submitted
    %}
    course_stat = courses(course_id,:);
    upper_bound = (1-(course_stat(3)*gr_0(course_id)/10)) * (34*course_stat(4)/(course_stat(1)+course_stat(2)));
end

% stores constant & slope of my linear functions
[c1,s1] = hr_to_gr(1, courses, gr_0, cred);
[c2,s2] = hr_to_gr(2, courses, gr_0, cred);
[c3,s3] = hr_to_gr(3, courses, gr_0, cred);
[c4,s4] = hr_to_gr(4, courses, gr_0, cred);

%%%%% linear programming %%%%%

sem_gpa_only = 0; % account for major gpa (or not)

if sem_gpa_only
    f = [1 1 1 1];
    A = [1 1 1 1; -s1 -s2 -s3 -s4];
    b = [idle_hr; -(sem_goal*sum(cred)-c1-c2-c3-c4)];
    
    vlb = [0 0 0 0];
    vub = [calc_ub(1,courses,gr_0) calc_ub(2,courses,gr_0) calc_ub(3,courses,gr_0) calc_ub(4,courses,gr_0)];
    
    display('semester gpa only')
    linprog(f,A,b,[],[],vlb,vub)

else
    % NOTE: assume that i am a math major
    f = [1 1 1 1];
    A = [1 1 1 1; -s1 -s2 -s3 -s4; -s1 -s2 0 0];
    b = [idle_hr; -(sem_goal*sum(cred)-c1-c2-c3-c4); -(maj_goal*(cred(1)+cred(2))-c1-c2)];

    vlb = [0 0 0 0];
    vub = [calc_ub(1,courses,gr_0) calc_ub(2,courses,gr_0) calc_ub(3,courses,gr_0) calc_ub(4,courses,gr_0)];

    display('semester & major gpa')
    linprog(f,A,b,[],[],vlb,vub)
end
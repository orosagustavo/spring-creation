module subrotinas
    implicit none
    
contains
    subroutine relative_vector(x1, y1, x2, y2, dx, dy)
        real, intent(in) :: x1, y1, x2, y2
        real, intent(out) ::  dx, dy

        dx = x1 - x2
        dy = y1 - y2
        
    end subroutine relative_vector

    subroutine spring_force(dx, dy, natural_distance, k_mol, fx, fy)
        implicit none
        real, intent(in) :: dx, dy, natural_distance, k_mol
        real :: particle_distance, delta_distance, force
        real, intent(out) ::  fx, fy

        particle_distance = sqrt(dx**2 + dy**2)
        delta_distance = particle_distance - natural_distance
        
        if ( particle_distance < 1.0e-6 ) then
            fx = 1.0e-6
            fy = 1.0e-6
        end if
        ! Calculate force
        force = -k_mol * delta_distance
        ! Normalized directional vector
        fx = force * (dx / particle_distance)
        fy = force * (dy / particle_distance)

    end subroutine spring_force

    subroutine damped_string_acc(vx, vy, fx, fy, b, mass, n, ax, ay)
        implicit none
        ! Entradas
        integer, intent(in) :: n
        real, dimension(n), intent(in) :: vx, vy, mass
        real, intent(in) :: fx, fy, b
        real, dimension(n), intent(out) :: ax, ay
        
        real, dimension(n) :: f_damps_x, f_damps_y
        integer :: i
    
        do i = 1, n
            ! Damping forces
            f_damps_x(i) = b * vx(i)
            f_damps_y(i) = b * vy(i)
        end do
    
        do i = 1, n
            ! Calculate acceleration for each particle
            if (i == 1) then
                ax(i) = (fx - f_damps_x(i)) / mass(i)
                ay(i) = (fy - f_damps_y(i)) / mass(i)
            else
                ax(i) = -(fx - f_damps_x(i)) / mass(i)
                ay(i) = -(fy - f_damps_y(i)) / mass(i)
            end if
        end do
    end subroutine damped_string_acc

    subroutine velocities(ax, ay, ax_0, ay_0, vx_0, vy_0, dt, n, vx, vy)
        implicit none

        integer, intent(in) :: n
        real, dimension(n), intent(in) :: ax, ay, ax_0, ay_0 
        real, dimension(n), intent(in) :: vx_0, vy_0
        real, intent(in) :: dt
        real, dimension(n), intent(out) :: vx, vy

        integer :: i

        do i = 1, n
            vx(i) = vx_0(i) + 0.5 * (ax(i) + ax_0(i)) * dt
            vy(i) = vy_0(i) + 0.5 * (ay(i) + ay_0(i)) * dt
        end do

    end subroutine velocities

    subroutine position(x_0, y_0, vx, vy, ax, ay, dt, n, x, y)
        implicit none

        integer, intent(in) :: n
        real, dimension(n), intent(in) :: x_0, y_0, vx, vy, ax, ay 
        real, intent(in) :: dt
        real, dimension(n), intent(out) :: x, y

        integer :: i

        do i = 1, n
            x(i) = x_0(i) + vx(i) * dt + 0.5 * ax(i) * dt**2
            y(i) = y_0(i) + vy(i) * dt + 0.5 * ay(i) * dt**2
        end do

    end subroutine position
        
end module subrotinas
